import torch
import torch.nn as nn

import math
from subnet import GetSubPatchFaster

class ERegister(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',
                 num_layers=1, use_prefix_tune_for_e_prompt=False, num_heads=-1, same_key_value=False,
                 sparse_patch_c=0.0, img_size=224, patch_size=16):
        super().__init__()

        self.length = length
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_layers = num_layers
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        self.num_heads = num_heads
        self.same_key_value = same_key_value
        self.embed_dim = embed_dim

        if self.prompt_pool:
            # user prefix style
            prompt_pool_shape = (self.pool_size, self.num_heads, embed_dim)
            self.register_tokens = nn.Parameter(torch.randn(prompt_pool_shape))
            nn.init.uniform_(self.register_tokens, -1, 1)
                    
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            self.register_key = nn.Parameter(torch.randn(key_shape))
            nn.init.uniform_(self.register_key, -1, 1)


        self.sparse_patch_c = sparse_patch_c
        if self.sparse_patch_c > 0:
            # patch selection
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) + 1
            self.p_m = nn.Parameter(torch.empty(num_patches))
            self.patch_mask = {}
            self.zeros_patch, self.ones_patch = torch.zeros(self.p_m.shape), torch.ones(self.p_m.shape)
            self.init_patch_parameters()

    def init_patch_parameters(self):

        for idx in range(self.pool_size):
            nn.init.kaiming_uniform_(self.p_m[None], a=math.sqrt(5))
            self.patch_mask[idx] = GetSubPatchFaster.apply(self.p_m.abs(),
                                                    self.zeros_patch,
                                                    self.ones_patch,
                                                    self.sparse_patch_c)
        


    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=None):
        out = dict()


        # positional_embedding
        if prompt_mask is not None and self.sparse_patch_c > 0:
            # patch selection
            self.patch_mask[prompt_mask[0].item()] = GetSubPatchFaster.apply(self.p_m.abs(),
                                                    self.zeros_patch,
                                                    self.ones_patch,
                                                    self.sparse_patch_c)

        if self.embedding_key == 'mean':
            x_embed_mean = torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'max':
            x_embed_mean = torch.max(x_embed, dim=1)[0]
        elif self.embedding_key == 'mean_max':
            x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
        elif self.embedding_key == 'cls':
            if cls_features is None:
                x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
            else:
                x_embed_mean = cls_features

        register_key_norm = self.l2_normalize(self.register_key, dim=-1) # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1) # B, C
        
        similarity = torch.matmul(register_key_norm, x_embed_norm.t()) # pool_size, B or Pool_size, #class, B
        similarity = similarity.t() # B, pool_size

        (similarity_top_k, idx) = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
        out['similarity'] = similarity

        if self.batchwise_prompt:
            register_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
            # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
            # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
            # Unless dimension is specified, this will be flattend if it is not already 1D.
            if register_id.shape[0] < self.pool_size:
                register_id = torch.cat([register_id, torch.full((self.pool_size - register_id.shape[0],), torch.min(idx.flatten()), device=register_id.device)])
                id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
            _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k

            if True:
                major_register_id = register_id[major_idx] # top_k
            else:
                major_register_id = torch.ones_like(register_id[major_idx]) * task_id # top_k 

            # expand to batch
            idx = major_register_id.expand(x_embed.shape[0], -1).contiguous() # B, top_k

        if prompt_mask is not None:
            idx = prompt_mask # B, top_k

            if self.sparse_patch_c > 0:
                patch_idx = torch.where(self.patch_mask[idx[0].item()] > 0)[0]
                x_embed = x_embed[:,patch_idx,:]
        
        out['register_idx'] = idx
        if self.use_prefix_tune_for_e_prompt:
            batched_register_raw = self.register_tokens[idx,:,]  # num_layers, B, top_k, length, C
            batch_size, top_k, num_heads, heads_embed_dim = batched_register_raw.shape
            batched_register = batched_register_raw.reshape(
                batch_size, top_k * num_heads, heads_embed_dim)[0]

        batched_key_norm = register_key_norm[idx] # B, top_k, C

        out['selected_key'] = batched_key_norm
        out['register_key_norm'] = register_key_norm
        out['x_embed_norm'] = x_embed_norm

        # Put pull_constraint loss calculation inside
        x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
        sim = batched_key_norm * x_embed_norm # B, top_k, C
        reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
        
        out['reduce_sim'] = reduce_sim
        out['batched_register'] = batched_register

        return out, x_embed
