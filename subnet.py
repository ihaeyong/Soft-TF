import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict
import math

import numpy as np
import numbers

from functools import partial
from helpers import to_2tuple

def percentile(scores, sparsity):
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubPatchFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):

        k_val = percentile(scores, sparsity * 100)
        masks = torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

        return masks

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class GetSubnetFasterMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):

        k_val = percentile(scores, sparsity * 100)
        masks = torch.where(scores < k_val,
                            zeros.to(scores.device),
                            ones.to(scores.device))
        return masks

    @staticmethod
    def backward(ctx, g):

        return g, None, g, None


class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):

        k_val = percentile(scores, sparsity * 100)
        masks = torch.where(scores < k_val,
                            zeros.to(scores.device),
                            ones.to(scores.device))
            
        return masks

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None    

    
class GetSubnetFasterTh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, k_val=0.0):

        # k_val = percentile(scores, sparsity * 100)
        masks = torch.where(scores < k_val,
                            zeros.to(scores.device),
                            ones.to(scores.device))
            
        return masks

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None
    
class GetSubnetFasterQKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores_q, scores_k, scores_v, zeros, ones, sparsity):


        k_val = percentile(scores_q, sparsity * 100)
        mask_q = torch.where(scores_q < k_val,
                            zeros.to(scores_q.device),
                            ones.to(scores_q.device))
        
        k_val = percentile(scores_k, sparsity * 100)
        mask_k = torch.where(scores_k < k_val,
                            zeros.to(scores_k.device),
                            ones.to(scores_k.device))
        
        k_val = percentile(scores_v, sparsity * 100)
        mask_v = torch.where(scores_v < k_val,
                            zeros.to(scores_v.device),
                            ones.to(scores_v.device))
        
        # masks = torch.cat((mask_q, mask_k, mask_v), dim=0)

        return mask_q, mask_k, mask_v

    @staticmethod
    def backward(ctx, q, k, v):
        return q, k, v, None, None, None


class GetSubnetGPM(torch.autograd.Function):
    def __init__(self):
        super(GetSubnetGPM, self).__init__()

    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity, x, weights, Uf):
        # dummy flag
        ctx.save_for_backward(torch.Tensor([True]).to(scores.device))

        if Uf is None and False:
            with torch.no_grad():
                # -- GPM ---
                bsz = x.size(0)
                b_idx = range(bsz)
                activation = torch.mm(x[b_idx,], weights.t()).t().cpu().numpy()
                U,S,Vh = np.linalg.svd(activation, full_matrices=False)

                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<0.999)
                feat=U[:,0:r]
                Uf=torch.Tensor(np.dot(feat,feat.transpose())).to(scores.device)
                ctx.save_for_backward(Uf)
        else:
            if Uf is not None:
                ctx.save_for_backward(Uf)

        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        # params.grad.data = params.grad.data - torch.mm(params.grad.data, Uf)
        if len(ctx.saved_tensors[0]) > 1:
            g = g - 0.01 * torch.mm(ctx.saved_tensors[0], g)
        else:
            None
        return g, None, None, None, None, None, None


## Define ResNet18 model
def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    return int(np.floor((Lin+2*padding-dilation*(kernel_size-1)-1)/float(stride)+1))

class GetSubnetGPMConv2d(torch.autograd.Function):
    def __init__(self):
        super(GetSubnetGPMConv2d, self).__init__()
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity, x, weights, stride, padding, Uf):

        # dummy flag
        ctx.save_for_backward(torch.Tensor([True]).to(scores.device))

        if False:
            with torch.no_grad():
                # -- GPM ---
                activation = F.conv2d(input=x, weight=weights, bias=None,stride=stride, padding=padding).cpu().numpy()
                # --------------------------
                out_ch, in_ch, ksz, ksz = weights.size()
                bsz, out_ch, sz, sz = activation.shape

                p1d = (1, 1, 1, 1)
                k = 0
                #sf = compute_conv_output_size(activation.shape, ksz, stride, padding)
                if False:
                    b_idx=sorted(np.random.choice(bsz, int(bsz*0.1),replace=False))
                else:
                    b_idx=range(bsz)
                mat = np.zeros((ksz*ksz*in_ch, sz*sz*len(b_idx)))
                act = F.pad(x, p1d, "constant", 0).detach().cpu().numpy()
                for kk in b_idxs:
                    for ii in range(sz):
                        for jj in range(sz):
                            mat[:,k]=act[kk,:,stride*ii:ksz+stride*ii,stride*jj:ksz+stride*jj].reshape(-1)
                            k +=1
                # activation
                U,S,Vh = np.linalg.svd(mat, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S**2).sum()
                sval_ratio = (S**2)/sval_total
                r = np.sum(np.cumsum(sval_ratio)<0.945)
                feat=U[:,0:r]
                Uf=torch.Tensor(np.dot(feat,feat.transpose())).to(scores.device)
                ctx.save_for_backward(Uf)
        else:
            if Uf is not None:
                ctx.save_for_backward(Uf)

        k_val = percentile(scores, sparsity*100)
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))


    @staticmethod
    def backward(ctx, g):
        # params.grad.data = params.grad.data - torch.mm(params.grad.data, Uf)
        sz = g.size(0)
        if len(ctx.saved_tensors[0]) > 1:
            g = g - 0.01 * torch.mm(g.view(sz, -1), ctx.saved_tensors[0]).view(g.size())
        else:
            None

        return g, None, None, None, None, None, None, None, None



class STEMult(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, m):
        ctx.save_for_backward(w)
        return w * m

    @staticmethod
    def backward(ctx, g):
        return g, g*ctx.saved_tensors[0].clone()

def get_none_masks(model):
        none_masks = {}
        for name, module in model.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                none_masks[name + '.weight'] = None
                none_masks[name + '.bias'] = None
                


class SupLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None, sparsity=0.01) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps=1e-5, elementwise_affine=True)

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
            
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.sparsity = sparsity
        self.num_tasks = 10
        
        if False:
            self.w_m = nn.ParameterList(
                [
                    nn.Parameter(self.mask_init(torch.empty(self.normalized_shape)))
                    for _ in range(self.num_tasks)
                ]
            )
            
        else:
            self.w_m = nn.Parameter(torch.empty(self.normalized_shape[0], self.normalized_shape[0]))
        
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if False:
            self.b_m = nn.ParameterList(
                [
                    nn.Parameter(self.mask_init(torch.empty(self.normalized_shape)))
                    for _ in range(self.num_tasks)
                ]
            )
            
        else:
            self.b_m = nn.Parameter(torch.empty(self.normalized_shape[0]))
        
        self.bias_mask = None
        self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)

        self.reset_parameters()
        self.init_mask_parameters(uniform=False)
            
        # warping modules
        if self.weight.ndim == 1:
            weight = self.weight[None]
            self.basis_coeff = nn.Parameter(torch.Tensor(weight.shape), requires_grad=True)
        elif self.weight.ndim == 2:
            weight = self.weight
            self.basis_coeff = nn.Parameter(torch.Tensor(self.weight.shape), requires_grad=True)
        elif self.weight.ndim != 2:
            weight = self.weight
            Co, Ci, k1, k2 = weight.shape
            self.basis_coeff = nn.Parameter(torch.Tensor(Co, Ci*k1*k2, 1, 1), requires_grad=True)
            self.register_buffer("UT_forward_conv", torch.Tensor(Ci*k1*k2, Ci, k1, k2))
            self.register_buffer("UT_backward_conv", torch.Tensor(Co, Co, 1, 1))
        
        # use register_buffer so model.to(device) works on fixed tensors like masks
        # self.register_buffer("forward_covariance", None)
        # self.register_buffer("basis_coefficients", torch.Tensor(weight.shape).reshape(weight.shape[0], -1))
        # self.register_buffer("coeff_mask", torch.zeros(self.basis_coeff.shape))
        # self.register_buffer("UT_forward", torch.eye(self.basis_coeff.shape[1]))
        # self.register_buffer("UT_backward", torch.eye(self.basis_coeff.shape[0]))
        
        self.coeff_mask = nn.ParameterList(
                [
                    nn.Parameter(self.mask_coeff_init(torch.empty(self.normalized_shape[0], self.normalized_shape[0])))
                    for _ in range(self.num_tasks)
                ]
            )

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        
    def mask_init(self, module):
        module = torch.ones_like(module) / self.normalized_shape[0]
        return module
    
    def mask_coeff_init(self, module):
        nn.init.normal_(module, math.sqrt(5))
        return module

    def init_mask_parameters(self, uniform=True):

        if uniform:
            #nn.init.ones_(self.w_m)
            self.w_m.data = torch.ones_like(self.w_m) / self.normalized_shape[0]
        else:
            nn.init.normal_(self.w_m, math.sqrt(5))

        if uniform:
            # nn.init.ones_(self.b_m)
            self.b_m.data = torch.ones_like(self.b_m) / self.normalized_shape[0]
        else:
            bound = 1 / math.sqrt(self.normalized_shape[0])
            nn.init.uniform_(self.b_m, -bound, bound)

        self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(), 
                                               self.zeros_weight,
                                               self.ones_weight,
                                               self.sparsity)
        
        if self.bias is not None:
            self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                    self.zeros_bias,
                                                    self.ones_bias,
                                                    self.sparsity)
        

        

    def forward(self, x, task_id, weight_mask=None, bias_mask=None, mode='train'):
        
        w_pruned, b_pruned = None, None
        beta = 1e-16
        # If training, Get the subnet by sorting the scores
        if mode=='train':
            if weight_mask is None:
                self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(),
                                                        self.zeros_weight,
                                                        self.ones_weight,
                                                        self.sparsity)
            else:
                self.weight_mask=weight_mask

            # deg = self.weight_mask.sum(1)
            # deg_inv_sqrt = deg.pow(-0.5) * beta
            w_pruned = self.weight[None] * self.weight_mask.diag()
            weight_mask = self.weight_mask
            # weight_mask[range(self.normalized_shape[0]),range(self.normalized_shape[0])] = 0
            w_pruned += self.weight[None] @ (weight_mask * self.coeff_mask[task_id] * beta)
            # w_pruned = self.weight * self.weight_mask.diag()
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                # deg = self.bias_mask.sum(1)
                # deg_inv_sqrt = deg.pow(-0.5) * beta
                # b_pruned = self.bias[None] * self.bias_mask.diag()
                # b_pruned += self.bias[None] @ ((self.bias_mask + 1e-3) * deg_inv_sqrt)
                # b_pruned = self.bias * self.coeff_mask[task_id].diag() * self.bias_mask.diag()
                b_pruned = self.bias * self.bias_mask

        elif mode=='valid':
            # deg = self.weight_mask.sum(1)
            # deg_inv_sqrt = deg.pow(-0.5) * beta
            w_pruned = self.weight[None] * self.weight_mask.diag()
            weight_mask = self.weight_mask
            # weight_mask[range(self.normalized_shape[0]),range(self.normalized_shape[0])] = 0
            w_pruned += self.weight[None] @ (weight_mask * self.coeff_mask[task_id] * beta) 
            # w_pruned = self.weight * self.weight_mask.diag()

            b_pruned = None
            if self.bias is not None:
                #b_pruned = self.bias_mask * self.bias  
                #deg = self.bias_mask.sum(1)
                #deg_inv_sqrt = deg.pow(-0.5) * beta
                #b_pruned = self.bias[None] * self.bias_mask.diag()
                # b_pruned += self.bias[None] @ ((self.bias_mask + 1e-3) * deg_inv_sqrt)
                # b_pruned = self.bias * self.coeff_mask[task_id].diag() * self.bias_mask.diag()
                b_pruned = self.bias * self.bias_mask

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            # deg = weight_mask.sum(1)
            # deg_inv_sqrt = deg.pow(-0.5) * beta
            # w_pruned = self.weight[None] * weight_mask.diag()
            # w_pruned += self.weight[None] @ ((weight_mask + 1e-3) * deg_inv_sqrt)
            w_pruned = self.weight[None] * weight_mask.diag()
            # weight_mask[range(self.normalized_shape[0]),range(self.normalized_shape[0])] = 0
            w_pruned += self.weight[None] @ (weight_mask * self.coeff_mask[task_id] * beta)
            
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias * bias_mask
        
        return F.layer_norm(x, self.normalized_shape, w_pruned[0], b_pruned, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SubLayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 bias=True, device=None, dtype=None, sparsity=0.01) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(normalized_shape, eps=1e-5, elementwise_affine=True)

        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.sparsity = sparsity
        
        self.w_m = nn.Parameter(torch.empty(self.normalized_shape))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        self.b_m = nn.Parameter(torch.empty(self.normalized_shape))
        self.bias_mask = None
        self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)

        self.reset_parameters()
        self.init_mask_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def init_mask_parameters(self, uniform=True):

        if uniform:
            #nn.init.ones_(self.w_m)
            self.w_m.data = torch.ones_like(self.w_m) / self.normalized_shape[0]
        else:
            nn.init.normal_(self.w_m, math.sqrt(5))

        if uniform:
            # nn.init.ones_(self.b_m)
            self.b_m.data = torch.ones_like(self.b_m) / self.normalized_shape[0]
        else:
            bound = 1 / math.sqrt(self.normalized_shape[0])
            nn.init.uniform_(self.b_m, -bound, bound)

        self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(),
                                                self.zeros_weight,
                                                self.ones_weight,
                                                self.sparsity)
        
        if self.bias is not None:
            self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                    self.zeros_bias,
                                                    self.ones_bias,
                                                    self.sparsity)

    def forward(self, x, weight_mask=None, bias_mask=None, mode='train'):
        
        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode=='train':
            if weight_mask is None:
                self.weight_mask=GetSubnetFaster.apply(self.w_m.abs(),
                                                       self.zeros_weight,
                                                       self.ones_weight,
                                                       self.sparsity)
            else:
                self.weight_mask = weight_mask

            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(),
                                                       self.zeros_bias,
                                                       self.ones_bias,
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias

        elif mode=='valid':
            w_pruned = self.weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.layer_norm(x, self.normalized_shape, w_pruned, b_pruned, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.01, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, 
                                             bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable
        self.batchwise_prompt = False

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = {}
        self.learn_mask = True
        self.num_tasks = 20

        if self.learn_mask is False:
            self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        else:
            self.ones_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features))
            self.zeros_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features))

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = {}
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        init_type = ['uniform', 'kaiming', 'xavier', 'normal']
        init_type_ = init_type[0]

        self.init_mask_parameters(init_type=init_type_)
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")
        
        self.get_masks(init_type=init_type_)

        self.start_task = 0
        self.num_tasks_learned = 0
        
    def get_masks(self, init_type):

        for subnet_id in range(self.num_tasks):
            self.init_mask_parameters(init_type=init_type)

            if self.learn_mask:
                zeros_weight = torch.sigmoid(self.zeros_weight)
                self.weight_mask[subnet_id] = GetSubnetFasterMask.apply(self.w_m.abs(),
                                                        zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity)
                
                self.bias_mask[subnet_id] = GetSubnetFasterMask.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        self.sparsity)
            else:
                self.weight_mask[subnet_id] = GetSubnetFaster.apply(self.w_m.abs(),
                                                        self.zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity)
                
                self.bias_mask[subnet_id] = GetSubnetFaster.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        self.sparsity) 
            
            
    @torch.no_grad()
    def cache_masks(self, device):

        m_mask = []
        for id in range(self.num_tasks_learned):
            m_mask.append(self.weight_mask[id].to(device))

        self.register_buffer(
            "stacked",
            torch.stack(m_mask).to(device),
        )


    def forward(self, x, weight_mask=None, bias_mask=None, mode='test', subnet_id=None):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if subnet_id==-2:
            mode = 'infer'
            self.cache_masks(device=x.device)
        
        if mode=='train':
            if subnet_id >= 0:
                if self.learn_mask:
                    zeros_weight = torch.sigmoid(self.zeros_weight)
                    self.weight_mask[subnet_id]=GetSubnetFasterMask.apply(self.w_m.abs(),
                                                        zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity)
                else:
                    self.weight_mask[subnet_id]=GetSubnetFaster.apply(self.w_m.abs(),
                                                        self.zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity) 

                w_pruned = self.weight_mask[subnet_id] * self.weight 
            else:
                w_pruned = self.weight

            b_pruned = None
            if subnet_id >= 0:

                if self.learn_mask:
                    self.bias_mask[subnet_id] = GetSubnetFasterMask.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        self.sparsity)
                else:
                    self.bias_mask[subnet_id] = GetSubnetFaster.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        self.sparsity) 

                b_pruned = self.bias_mask[subnet_id] * self.bias
            else:
                b_pruned = self.bias

        elif mode=='valid':

            if not self.batchwise_prompt:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias


            else:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "infer":
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned-self.start_task]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned-self.start_task)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)

            #import ipdb; ipdb.set_trace()
            weight_mask = (
                alpha_weights[idxs]
                * self.stacked[self.start_task: self.num_tasks_learned][idxs]
            ).sum(dim=0)

            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)

    def init_mask_parameters(self, uniform=True, init_type='uniform'):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
            if self.learn_mask:
                nn.init.uniform_(self.zeros_weight, a=-50, b=-10)
                
                if init_type == 'uniform':
                    nn.init.uniform_(self.ones_weight, a=0.99, b=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(self.ones_weight, a=math.sqrt(5))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(self.ones_weight)
                elif init_type == 'normal':
                    nn.init.normal_(self.ones_weight, mean=0.0, std=0.01)
        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SubnetLinearV1(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.01, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, 
                                             bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable
        self.batchwise_prompt = False

        # Mask Parameters of Weights and Bias
        self.weight_mask = {}
        self.learn_mask = True
        self.num_tasks = 20

        self.ones_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features))

        if bias:
            self.ones_bias = nn.Parameter(torch.empty(out_features))
            self.bias_mask = {}
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        init_type = ['uniform', 'kaiming', 'xavier', 'normal']
        init_type_ = init_type[0]

        self.init_mask_parameters(init_type=init_type_)
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")
        
        self.get_masks(init_type=init_type_)

        self.start_task = 0
        self.num_tasks_learned = 0
        
    def get_masks(self, init_type):

        for subnet_id in range(self.num_tasks):
            self.init_mask_parameters(init_type=init_type)
            self.weight_mask[subnet_id] = self.ones_weight[subnet_id]
            #self.bias_mask[subnet_id] = self.b_m
            
    @torch.no_grad()
    def cache_masks(self, device):

        m_mask = []
        for id in range(self.num_tasks_learned):
            m_mask.append(self.weight_mask[id].to(device))

        self.register_buffer(
            "stacked",
            torch.stack(m_mask).to(device),
        )


    def forward(self, x, weight_mask=None, bias_mask=None, mode='test', subnet_id=None):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if subnet_id==-2:
            mode = 'infer'
            self.cache_masks(device=x.device)
        
        if mode=='train':
            if subnet_id >= 0:
                self.weight_mask[subnet_id] = self.ones_weight[subnet_id]
                w_pruned = self.weight_mask[subnet_id] * self.weight 
            else:
                w_pruned = self.weight

            b_pruned = None
            if subnet_id >= 0:
                self.bias_mask[subnet_id] = self.ones_bias
                b_pruned = self.bias_mask[subnet_id] * self.bias
            else:
                b_pruned = self.bias

        elif mode=='valid':

            if not self.batchwise_prompt:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias


            else:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "infer":
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned-self.start_task]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned-self.start_task)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)

            #import ipdb; ipdb.set_trace()
            weight_mask = (
                alpha_weights[idxs]
                * self.stacked[self.start_task: self.num_tasks_learned][idxs]
            ).sum(dim=0)

            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self, uniform=True, init_type='uniform'):
        if uniform:
            # nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
            if self.learn_mask:
                #nn.init.uniform_(self.zeros_weight, a=-50, b=-10)
                if init_type == 'uniform':
                    nn.init.uniform_(self.ones_weight, a=0.99, b=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(self.ones_weight, a=math.sqrt(5))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(self.ones_weight)
                elif init_type == 'normal':
                    nn.init.normal_(self.ones_weight, mean=0.0, std=0.01)
        else:
            pass 
            #nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.ones_weight[0])
            bound = 1 / math.sqrt(fan_in)
            #nn.init.uniform_(self.b_m, -bound, bound)


class SubnetLinearQKV(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.01, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, 
                                             bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable
        self.batchwise_prompt = False

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = {}
        self.learn_mask = True
        self.num_tasks = 20

        if self.learn_mask is False:
            self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        else:
            self.ones_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features)) # torch.ones(self.w_m.shape)
            self.zeros_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features))

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = {}
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        init_type = ['uniform', 'kaiming', 'xavier', 'normal']
        init_type_ = init_type[0]

        self.init_mask_parameters(init_type=init_type_)
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")
        
        self.get_masks(init_type=init_type_)

        self.start_task = 0
        self.num_tasks_learned = 0
        
    def get_masks(self, init_type):

        for subnet_id in range(self.num_tasks):
            self.init_mask_parameters(init_type=init_type)

            if self.learn_mask:
                zeros_weight = torch.sigmoid(self.zeros_weight)
                self.weight_mask[subnet_id] = GetSubnetFasterMask.apply(self.w_m.abs(),
                                                        zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity)
                
                self.bias_mask[subnet_id] = GetSubnetFasterMask.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        0.0)
            else:
                self.weight_mask[subnet_id] = GetSubnetFaster.apply(self.w_m.abs(),
                                                        self.zeros_weight[subnet_id],
                                                        self.ones_weight[subnet_id],
                                                        self.sparsity)
                
                self.bias_mask[subnet_id] = GetSubnetFaster.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        0.0)
            
    @torch.no_grad()
    def cache_masks(self, device):

        m_mask = []
        for id in range(self.num_tasks_learned):
            m_mask.append(self.weight_mask[id].to(device))

        self.register_buffer(
            "stacked",
            torch.stack(m_mask).to(device),
        )

    def forward(self, x, weight_mask=None, bias_mask=None, mode='test', subnet_id=None):

        w_pruned, b_pruned = None, None

        if subnet_id==-2:
            mode = 'infer'
            self.cache_masks(device=x.device)

        # If training, Get the subnet by sorting the scores
        if mode=='train':
            if subnet_id >= 0:
                if self.learn_mask:
                    zeros_weight = torch.sigmoid(self.zeros_weight)
                    self.weight_mask[subnet_id]=GetSubnetFasterMask.apply(self.w_m.abs(),
                                                    zeros_weight[subnet_id],
                                                    self.ones_weight[subnet_id],
                                                    self.sparsity)
                else:
                    self.weight_mask[subnet_id]=GetSubnetFaster.apply(self.w_m.abs(),
                                                    self.zeros_weight[subnet_id],
                                                    self.ones_weight[subnet_id],
                                                    self.sparsity) 

                w_pruned = self.weight_mask[subnet_id] * self.weight 
            else:
                w_pruned = self.weight

            b_pruned = None
            if subnet_id >= 0:
                if self.learn_mask:
                    self.bias_mask[subnet_id] = GetSubnetFasterMask.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        0.0)
                else:
                    self.bias_mask[subnet_id] = GetSubnetFaster.apply(self.b_m.abs(),
                                                        self.zeros_bias,
                                                        self.ones_bias,
                                                        0.0) 
                
                b_pruned = self.bias_mask[subnet_id] * self.bias
            else:
                b_pruned = self.bias

        elif mode=='valid':

            if not self.batchwise_prompt:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias 
            else:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "infer":
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned-self.start_task]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned-self.start_task)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)

            weight_mask = (
                alpha_weights[idxs]
                * self.stacked[self.start_task: self.num_tasks_learned][idxs]
            ).sum(dim=0)

            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)

    def init_mask_parameters(self, uniform=True, init_type='uniform'):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))

            if self.learn_mask:
                nn.init.uniform_(self.zeros_weight, a=-50, b=-10.0)

                if init_type == 'uniform':
                    nn.init.uniform_(self.ones_weight, a=0.99, b=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(self.ones_weight, a=math.sqrt(5))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(self.ones_weight)
                elif init_type == 'normal':
                    nn.init.normal_(self.ones_weight, mean=0.0, std=0.01)

        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

class SubnetLinearQKVV1(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.01, trainable=True):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, 
                                             bias=bias)
        self.sparsity = sparsity
        self.trainable = trainable
        self.batchwise_prompt = False

        # Mask Parameters of Weights and Bias
        self.weight_mask = {}
        self.learn_mask = True
        self.num_tasks = 20

        self.ones_weight = nn.Parameter(torch.empty(self.num_tasks, out_features, in_features))

        if bias:
            self.ones_bias = nn.Parameter(torch.empty(out_features))
            self.bias_mask = {}
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        init_type = ['uniform', 'kaiming', 'xavier', 'normal']
        init_type_ = init_type[0]

        self.init_mask_parameters(init_type=init_type_)
        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")
        
        self.get_masks(init_type=init_type_)

        self.start_task = 0
        self.num_tasks_learned = 0
        
    def get_masks(self, init_type):

        for subnet_id in range(self.num_tasks):
            self.init_mask_parameters(init_type=init_type)
            self.weight_mask[subnet_id] = self.ones_weight[subnet_id]
            #self.bias_mask[subnet_id] = self.ones_bias
            
    @torch.no_grad()
    def cache_masks(self, device):

        m_mask = []
        for id in range(self.num_tasks_learned):
            m_mask.append(self.weight_mask[id].to(device))

        self.register_buffer(
            "stacked",
            torch.stack(m_mask).to(device),
        )

    def forward(self, x, weight_mask=None, bias_mask=None, mode='test', subnet_id=None):

        w_pruned, b_pruned = None, None

        if subnet_id==-2:
            mode = 'infer'
            self.cache_masks(device=x.device)

        # If training, Get the subnet by sorting the scores
        if mode=='train':
            if subnet_id >= 0:
                self.weight_mask[subnet_id] = self.ones_weight[subnet_id]
                w_pruned = self.weight_mask[subnet_id] * self.weight 
            else:
                w_pruned = self.weight

            b_pruned = None
            if subnet_id >= 0:
                self.bias_mask[subnet_id] = self.ones_bias
                b_pruned = self.bias_mask[subnet_id] * self.bias
            else:
                b_pruned = self.bias

        elif mode=='valid':
            if not self.batchwise_prompt:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias 
            else:
                if subnet_id == -1:
                    w_pruned = self.weight
                else:
                    w_pruned = self.weight_mask[subnet_id].to(x.device) * self.weight

                b_pruned = None
                if self.bias is not None:
                    if subnet_id == -1:
                        b_pruned = self.bias
                    else:
                        b_pruned = self.bias_mask[subnet_id].to(x.device) * self.bias

        # If inference, no need to compute the subnetwork
        elif mode=='test':
            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "infer":
            # Superimposed forward pass
            alpha_weights = self.alphas[: self.num_tasks_learned-self.start_task]
            idxs = (alpha_weights > 0).squeeze().view(self.num_tasks_learned-self.start_task)
            if len(idxs.shape) == 0:
                idxs = idxs.view(1)

            weight_mask = (
                alpha_weights[idxs]
                * self.stacked[self.start_task: self.num_tasks_learned][idxs]
            ).sum(dim=0)

            w_pruned = weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)

    def init_mask_parameters(self, uniform=True, init_type='uniform'):
        if uniform:
            # nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
            if self.learn_mask:
                #nn.init.uniform_(self.zeros_weight, a=-50, b=-10.0)
                if init_type == 'uniform':
                    nn.init.uniform_(self.ones_weight, a=0.99, b=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(self.ones_weight, a=math.sqrt(5))
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(self.ones_weight)
                elif init_type == 'normal':
                    nn.init.normal_(self.ones_weight, mean=0.0, std=0.01)

        else:
            pass
            #nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.ones_weight[0])
            bound = 1 / math.sqrt(fan_in)
            #nn.init.uniform_(self.b_m, -bound, bound)


class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.01, trainable=True):

        super(self.__class__, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity
        self.trainable = trainable

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)

        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        # Init Mask Parameters
        self.init_mask_parameters()

        if trainable == False:
            raise Exception("Non-trainable version is not yet implemented")

    def forward(self, x, weight_mask=None, bias_mask=None, mode='test'):

        w_pruned, b_pruned = None, None
        # If training, Get the subnet by sorting the scores
        if mode == 'train':
            self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), 
                                                       self.zeros_bias, 
                                                       self.ones_bias, 
                                                       self.sparsity)
                b_pruned = self.bias_mask * self.bias

        elif mode=='valid':
            w_pruned = self.weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == 'test':
            w_pruned = weight_mask * self.weight

            b_pruned = None
            if self.bias is not None:
                b_pruned = bias_mask * self.bias

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def update_w_m(self):
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)

    def init_mask_parameters(self, uniform=True):
        if uniform:
            nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        else:
            nn.init.normal_(self.w_m, std)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)

        # init weight scores
        # self.update_w_m()
        self.weight_mask = GetSubnetFaster.apply(self.w_m.abs(),
                                                     self.zeros_weight,
                                                     self.ones_weight,
                                                     self.sparsity)

        if self.bias is not None:
            self.bias_mask = GetSubnetFaster.apply(self.b_m.abs(), 
                                                   self.zeros_bias, 
                                                   self.ones_bias, 
                                                   self.sparsity)
            

class SubnetMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            sparsity=0.01,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(SubnetConv2d, kernel_size=1) if use_conv else SubnetLinearV1
        nn_linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1_subnet = True

        if self.fc1_subnet:
            self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0], sparsity=sparsity)
        else:
            self.fc1 = nn_linear_layer(in_features, hidden_features, bias=bias[0]) 
        
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()

        self.fc2_subnet = True

        if self.fc2_subnet:        
            self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1], sparsity=sparsity)
        else:
            self.fc2 = nn_linear_layer(hidden_features, out_features, bias=bias[1])
        
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, prompt_idx=None):
        
        # if prompt_idx is not None:
        mode = 'train' if self.training else 'valid'
        subnet_id = prompt_idx.item() if prompt_idx is not None else 0


        if self.fc1_subnet:
            x = self.fc1(x, mode=mode, subnet_id=subnet_id)
        else:
            x = self.fc1(x)

        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        if self.fc2_subnet:
            x = self.fc2(x, mode=mode, subnet_id=subnet_id)
        else:
            x = self.fc2(x)

        x = self.drop2(x)
        return x


def get_nonlin_func(nonlin):
    if nonlin == "tanh":
        return torch.tanh
    elif nonlin == "relu":
        return torch.relu
    elif nonlin == "gelu":
        return nn.functional.gelu
    elif nonlin == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError("Unsupported nonlinearity!")
    
class AdapterLayer(nn.Module):
    def __init__(self, dim, adapter_latent_size, adapter_residual=True):
        super().__init__()

        self.adapter_input_size = dim
        self.adapter_latent_size = adapter_latent_size
        self.non_linearity = get_nonlin_func(nonlin="gelu")
        self.residual = adapter_residual
        self.num_tasks= 20

        # down projection
        self.adapter_down_proj = nn.Parameter(torch.empty(self.num_tasks, self.adapter_latent_size, self.adapter_input_size))
        # up projection
        self.adapter_up_proj = nn.Parameter(torch.empty(self.num_tasks, self.adapter_input_size, self.adapter_latent_size))

        self.init_weights()

    def init_weights(self):
        """ Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function """
        nn.init.kaiming_uniform_(self.adapter_down_proj, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.adapter_up_proj, a=math.sqrt(5))

    def forward(self, x, mode='train', subnet_id=None):

        output = F.linear(input=x, weight=self.adapter_down_proj[subnet_id], bias=None)
        output = self.non_linearity(output)
        output = F.linear(input=output, weight=self.adapter_up_proj[subnet_id], bias=None)

        if self.residual:
            output = x + output
        return output
    
class AdaptMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            sparsity=0.01,
            param_ratio=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        nn_linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        hidden_dim_1 = param_ratio #32 
        hidden_dim_2 = param_ratio #32

        self.fc1 = nn_linear_layer(in_features, hidden_features, bias=bias[0])
        self.adapter1 = AdapterLayer(dim=hidden_features, adapter_latent_size=hidden_dim_1)
        
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()

        self.fc2 = nn_linear_layer(hidden_features, out_features, bias=bias[1])
        self.adapter2 = AdapterLayer(dim=out_features, adapter_latent_size=hidden_dim_2)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, prompt_idx=None):
        
        # if prompt_idx is not None:
        mode = 'train' if self.training else 'valid'
        subnet_id = prompt_idx.item() if prompt_idx is not None else 0
        x = self.fc1(x)
        x = self.adapter1(x, mode=mode, subnet_id=subnet_id)

        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)

        x = self.fc2(x)
        x = self.adapter2(x, mode=mode, subnet_id=subnet_id)
        x = self.drop2(x)

        return x