# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
import sys
import argparse
import datetime
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn

from pathlib import Path

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from datasets import build_continual_dataloader
from engine import *
import models
import utils

import wandb

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')


import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt


def density_estimate(dir_path, data1, data2, data3):

    # Generate sample data
    #np.random.seed(42)
    #data = np.random.randn(1000)  # Normally distributed data

    min_val = round(np.concatenate([data1, data2, data3]).min())
    max_val = round(np.concatenate([data1, data2, data3]).max()) 

    # Reshape data for KDE
    data1 = data1[:, np.newaxis]
    data2 = data2[:, np.newaxis]
    data3 = data3[:, np.newaxis]

    # Perform KDE
    kde1 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data1)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data2)
    kde3 = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(data3)    


    # Generate a range of values for estimation
    x_d = np.linspace(-4, 4, 1000)[:, np.newaxis]

    # Estimate density
    log_density1 = kde1.score_samples(x_d)
    density1 = np.exp(log_density1)

    log_density2 = kde2.score_samples(x_d)
    density2 = np.exp(log_density2)

    log_density3 = kde3.score_samples(x_d)
    density3 = np.exp(log_density3)

    # Plot the estimated density
    fig = plt.figure(figsize=(8,8))
    plt.plot(x_d[:, 0], density1, label='qvk1', color='r')
    plt.fill_between(x_d[:, 0], density1, alpha=0.2, color='r')

    plt.plot(x_d[:, 0], density2, label='qvk2', color='g')
    plt.fill_between(x_d[:, 0], density2, alpha=0.2, color='g')

    plt.plot(x_d[:, 0], density3, label='qvk3', color='b')
    plt.fill_between(x_d[:, 0], density3, alpha=0.2, color='b')
    
    #plt.title("Kernel Density Estimation")
    plt.xlabel("Data")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    fig.savefig(dir_path + '/kde.pdf', dpi=fig.dpi, bbox_inches='tight')


def hist(dir_path, data1, data2, data3, module, layer):

    # Estimate density using a histogram
    fig, ax = plt.subplots()
    #density, bins, _ = plt.hist(data1, bins=300, density=True, alpha=0.5, color='r', label='{}_L[{}]_Task{}'.format(module, layer, 1))

    density, bins, _ = plt.hist(data2, bins=300, density=True, alpha=0.5, color='g', label='{}_L[{}]_Task{}'.format(module, layer, 2))

    density, bins, _ = plt.hist(data3, bins=300, density=True, alpha=0.3, color='b', label='{}_L[{}]_Task{}'.format(module, layer, 3))

    # Plot the histogram
    # plt.title("Histogram Density Estimation")
    plt.xlabel("Parameters", fontsize=26)
    plt.ylabel("Density", fontsize=26)

    plt.xlim([0.9, 1.1])      # X축의 범위: [xmin, xmax]
    
    if 'QKV' in module:
        plt.ylim([2.3, 7.0])     # Y축의 범위: [ymin, ymax]

    else:
        plt.ylim([2.3, 7.0])     # Y축의 범위: [ymin, ymax]

    every_nth = 4
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    every_nth = 2
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    
    plt.setp(ax.get_xticklabels(), fontsize=26)
    plt.setp(ax.get_yticklabels(), fontsize=26)
    
    plt.legend(fontsize=16, loc='upper left')
    plt.grid()
    plt.show()

    fig.savefig(dir_path + '/{}_L[{}].pdf'.format(module, layer), dpi=fig.dpi, bbox_inches='tight')
    plt.close()


def hist_diff(dir_path, weight, data, module, layer):

    # Estimate density using a histogram
    fig, ax = plt.subplots()
    density, bins, _ = plt.hist(weight, bins=300, density=True, alpha=0.5, color='m', label='{}_L[{}]_Pre-trained Weight'.format(module, layer))

    density, bins, _ = plt.hist(data, bins=300, density=True, alpha=0.3, color='g', label='{}_L[{}]_Task{}'.format(module, layer, 2))

    # Plot the histogram
    # plt.title("Histogram Density Estimation")
    plt.xlabel("Parameters", fontsize=26)
    plt.ylabel("Density", fontsize=26)

    plt.xlim([-0.2, 0.2])      # X축의 범위: [xmin, xmax]
    
    plt.ylim([2.3, 7.0])     # Y축의 범위: [ymin, ymax]

    every_nth = 4
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    every_nth = 2
    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

    
    plt.setp(ax.get_xticklabels(), fontsize=26)
    plt.setp(ax.get_yticklabels(), fontsize=26)
    
    plt.legend(fontsize=16, loc='upper left')
    plt.grid()
    plt.show()

    fig.savefig(dir_path + '/diff_{}_L[{}].pdf'.format(module, layer), dpi=fig.dpi, bbox_inches='tight')
    plt.close()


def main(args):
    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True
    data_loader, class_mask = build_continual_dataloader(args)

    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        target_layers=args.target_layers
    )
    
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=args.prompt_pool,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
        use_prompt_mask=args.use_prompt_mask,
        use_g_prompt=args.use_g_prompt,
        g_prompt_length=args.g_prompt_length,
        g_prompt_layer_idx=args.g_prompt_layer_idx,
        use_prefix_tune_for_g_prompt=args.use_prefix_tune_for_g_prompt,
        use_e_prompt=args.use_e_prompt,
        e_prompt_layer_idx=args.e_prompt_layer_idx,
        use_prefix_tune_for_e_prompt=args.use_prefix_tune_for_e_prompt,
        same_key_value=args.same_key_value,
        sparse_patch_c = args.sparse_patch_c,
        num_register_tokens = args.num_register_tokens,
        sparsity = args.sparsity,
        use_lora = args.use_lora,
        target_layers = args.target_layers,
        subnet = args.subnet,
        param_ratio = args.param_ratio,
    )
    original_model.to(device)
    model.to(device)

    if args.freeze:

        org_num_params = 0
        soft_num_params = 0
        # all parameters are frozen for original vit model
        for n, p in original_model.named_parameters():
            p.requires_grad = False
            org_num_params += torch.numel(p)
            print('# params_{}: origin_{}'.format(n, org_num_params))

        model_num_params = 0
        prompt_num_params = 0
        soft_num_params = 0
        lora_num_params = 0
        adapt_num_params = 0
        for n, p in model.named_parameters():
            if 'ones_weight' in n:
                soft_num_params += torch.numel(p[0])
            elif 'lora_' in n:
                lora_num_params += torch.numel(p[0])
            elif 'adapter_' in n:
                adapt_num_params += torch.numel(p[0])
            elif 'prompt_' in n:
                prompt_num_params += torch.numel(p[0])
            else:
                model_num_params += torch.numel(p)

            print('# params_{}: model_{}, prompt_{}, soft_{}, lora_{}, adapt_{}'.format(n, model_num_params, prompt_num_params, soft_num_params, lora_num_params, adapt_num_params))

        if args.subnet == 'soft':
            num_params = soft_num_params
        elif args.subnet == 'lora':
            num_params = lora_num_params
        elif args.subnet == 'adapter':
            num_params = adapt_num_params
        else:
            num_params = 0

        wandb.log({'train/num_prompt': prompt_num_params, 'train/num_params': num_params})
        
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        for n, p in model.named_parameters():
            # n.startswith(tuple(['blocks','patch_embed', 'cls_token', 'norm', 'pos_embed']))
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False 

            # for subnetworks
            if n.startswith(tuple(['blocks'])):
                if 'ones_weight' in n or 'zeros_weight' in n or 'w_m' in n:
                    #if 'ones_weight' in n or 'w_m' in n:  
                    p.requires_grad = True
                elif 'lora_' in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            if p.requires_grad:
                print(n, p.requires_grad)
    
    print(args)

    if args.eval:
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):

            if task_id == args.num_tasks - 1:
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
                if os.path.exists(checkpoint_path):
                    print('Loading checkpoint from:', checkpoint_path)
                    checkpoint = torch.load(checkpoint_path)
                    model.load_state_dict(checkpoint['model'], False)
                else:
                    print('No checkpoint found at:', checkpoint_path)
                    return

                if False:
                    for layer in [9,10,11]:
                        # QKV
                        module = 'QKV'
                        data1 = (model.blocks[11].attn.qkv.ones_weight[0]).view(-1).detach().cpu().numpy()
                        data2 = (model.blocks[11].attn.qkv.ones_weight[1]).view(-1).detach().cpu().numpy()
                        data3 = (model.blocks[11].attn.qkv.ones_weight[2]).view(-1).detach().cpu().numpy() 
                            
                        hist(args.output_dir, data1, data2, data3, module=module, layer=layer+1)

                        weight = model.blocks[11].attn.qkv.weight
                        p_data = (weight * model.blocks[11].attn.qkv.ones_weight[0]).view(-1).detach().cpu().numpy() 

                        hist_diff(args.output_dir, weight.view(-1).detach().cpu().numpy(), p_data, module=module, layer=layer+1)

                        # proj
                        module = 'PROJ'
                        data1 = (model.blocks[11].attn.proj.ones_weight[0]).view(-1).detach().cpu().numpy()
                        data2 = (model.blocks[11].attn.proj.ones_weight[1]).view(-1).detach().cpu().numpy()
                        data3 = (model.blocks[11].attn.proj.ones_weight[2]).view(-1).detach().cpu().numpy()

                        hist(args.output_dir, data1, data2, data3, module=module, layer=layer+1)

                        weight = model.blocks[11].attn.proj.weight
                        p_data = (weight * model.blocks[11].attn.proj.ones_weight[1]).view(-1).detach().cpu().numpy() 

                        hist_diff(args.output_dir, weight.view(-1).detach().cpu().numpy(), p_data, module=module, layer=layer+1)

                        # mlp.fc1
                        module = 'MLP.FC1'
                        data1 = (model.blocks[11].mlp.fc1.ones_weight[0]).view(-1).detach().cpu().numpy()
                        data2 = (model.blocks[11].mlp.fc1.ones_weight[1]).view(-1).detach().cpu().numpy()
                        data3 = (model.blocks[11].mlp.fc1.ones_weight[2]).view(-1).detach().cpu().numpy()

                        hist(args.output_dir, data1, data2, data3, module=module, layer=layer+1)

                        weight = model.blocks[11].mlp.fc1.weight
                        p_data = (weight * model.blocks[11].mlp.fc1.ones_weight[1]).view(-1).detach().cpu().numpy() 

                        hist_diff(args.output_dir, weight.view(-1).detach().cpu().numpy(), p_data, module=module, layer=layer+1)

                        module = 'MLP.FC2'
                        data1 = (model.blocks[11].mlp.fc2.ones_weight[0]).view(-1).detach().cpu().numpy()
                        data2 = (model.blocks[11].mlp.fc2.ones_weight[1]).view(-1).detach().cpu().numpy()
                        data3 = (model.blocks[11].mlp.fc2.ones_weight[2]).view(-1).detach().cpu().numpy()

                        hist(args.output_dir, data1, data2, data3, module=module, layer=layer+1)


                        weight = model.blocks[11].mlp.fc2.weight
                        p_data = (weight * model.blocks[11].mlp.fc2.ones_weight[1]).view(-1).detach().cpu().numpy() 

                        hist_diff(args.output_dir, weight.view(-1).detach().cpu().numpy(), p_data, module=module, layer=layer+1)


                _ = evaluate_till_now(model, original_model, data_loader, device, 
                                                task_id, class_mask, acc_matrix, args,)
        
        return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if args.unscale_lr:
        global_batch_size = args.batch_size
    else:
        global_batch_size = args.batch_size * args.world_size
    args.lr = args.lr * global_batch_size / 256.0

    optimizer = create_optimizer(args, model_without_ddp)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    train_and_evaluate(model, model_without_ddp, original_model,
                       criterion, data_loader, optimizer, lr_scheduler,
                       device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    config = parser.parse_known_args()[-1][0]
    subparser = parser.add_subparsers(dest='subparser_name')

    if config == '10cifar100_dualprompt_pgp':
        from configs.cifar100_10_dualprompt_pgp import get_args_parser
        config_parser = subparser.add_parser('10cifar100_dualprompt_pgp', help='10-Split-CIFAR100 DualPrompt-PGP configs')
    elif config == '20cifar100_dualprompt_pgp':
        from configs.cifar100_20_dualprompt_pgp import get_args_parser
        config_parser = subparser.add_parser('20cifar100_dualprompt_pgp', help='20-Split-CIFAR100 DualPrompt-PGP configs')
    elif config == 'imr_dualprompt_pgp':
        from configs.imr_dualprompt_pgp import get_args_parser
        config_parser = subparser.add_parser('imr_dualprompt_pgp', help='Split-ImageNet-R DualPrompt-PGP configs')
    elif config == 'tinyimagenet_dualprompt_pgp':
        from configs.tinyimagenet_dualprompt_pgp import get_args_parser
        config_parser = subparser.add_parser('tinyimagenet_dualprompt_pgp', help='Split-TinyImagenet DualPrompt-PGP configs')
    elif config == 'cub200_dualprompt_pgp':
        from configs.cub200_dualprompt_pgp import get_args_parser
        config_parser = subparser.add_parser('cub200_dualprompt_pgp', help='Split-CUB200 DualPrompt-PGP configs')
    else:
        raise NotImplementedError
        
    get_args_parser(config_parser)

    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    exp_name = args.model
    exp_name += '_' + args.subnet 
    exp_name += '_prompt_length{}_size{}'.format(args.length, args.size)

    if args.subnet in ['lora', 'adapter', 'soft']:
        exp_name += '_target_layer' + ''.join(str(x) for x in args.target_layers)

        if args.subnet in ['lora', 'adapter']:
            exp_name += f'_param_r{args.param_ratio}'

    if args.sparse_patch_c > 0:
       exp_name += '_sparse_pc{}'.format(args.sparse_patch_c)
       args.sparse_patch_c = 1 - args.sparse_patch_c 

    if args.num_register_tokens > 0:
        exp_name += '_reg{}'.format(args.num_register_tokens)

    #if args.sparsity > 0:
    #   exp_name += '_sparsity{}'.format(args.sparsity)
    #   args.sparsity = 1.0 - args.sparsity
    #   exp_name += '_soft_layer' + ''.join(str(x) for x in args.e_prompt_layer_idx)

    exp_name += '_{}'.format(args.infer_type)

    if args.batchwise_prompt:
        exp_name += '_batch_prompt'

    if args.no_pgp:
        exp_name += '_no_pgp'

    if args.g_perturb_type != 'None':
        exp_name += f'_{args.g_perturb_type}_n{args.noise_scale}_rho{args.rho_max}_{args.rho_min}_s{1-args.sparsity_grad}'

    if args.output_dir:
        args.output_dir += '/' + exp_name
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.config = config
    wandb.init(project='{}'.format(config),
               entity='haeyong', name=exp_name, config=args)
    
    main(args)
    
    sys.exit(0)