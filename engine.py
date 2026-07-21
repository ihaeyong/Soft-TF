# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for dualprompt implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np
from sklearn.decomposition import PCA

from timm.utils import accuracy
from timm.optim import create_optimizer

import utils
import memory
import wandb

import torch.autograd as autograd

from subnet import percentile

import time


def grad_perturb(param, ones_weight, task_id, args):

    # Gradient Noise Injection (GNI)
    if args.g_perturb_type == 'GNI':
        noise_scale = 0.01
        ones_grad += noise_scale * torch.randn_like(param) 

    # Masked Gradient Noise Injection (MGNI)
    elif args.g_perturb_type == 'MGNI':
        noise_scale = 0.01
        ones_grad[param > 0.99] += noise_scale * torch.randn_like(param)[param > 0.99] 

    elif args.g_perturb_type == 'MGaussian':
        noise_scale = 0.01
        ones_grad[param > 0.99] += noise_scale * torch.normal(mean=0.0, std=1.0, size=param.size()).to(param.device)[param > 0.99]

    elif args.g_perturb_type == 'MLangevinDynamics':
        eta = 0.01
        sigma = 0.1
        noise_scale = math.sqrt(2 * eta * sigma)
        ones_grad[param > 0.99] += noise_scale * torch.normal(mean=0.0, std=sigma, size=param.size()).to(param.device)[param > 0.99]

    elif args.g_perturb_type == 'MBall':

        sparsity = 1 - args.sparsity_grad
        k_val = percentile(param, sparsity * 100)
        s_max = param.max()
        s_min = param.min()

        rho_i = args.rho_max - ((param - s_min) / (s_max - s_min)) * (args.rho_max - args.rho_min)
        rho_i *= args.noise_scale

        rho = torch.zeros_like(ones_weight)

        rho[task_id] = rho_i * (param >= k_val).float()
        ones_weight.grad += rho
    
    else:
        pass


def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, feature_prefix_gt, feature_prefix_et,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'
    
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        prompt_id = output['prompt_idx'][0][0]

        # here is the trick to mask out classes of non-current tasks
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

        loss = criterion(logits, target) # base criterion (CrossEntropyLoss)
        if args.pull_constraint and 'reduce_sim' in output:
            loss = loss - args.pull_constraint_coeff * output['reduce_sim']

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        # Grad Projection
        # freeze args.freeze[blocks, patch_embed, cls_token] parameters
        #if task_id != 0 and not args.no_pgp: 
        if True:
            grad_list = []
            for k, (m, params) in enumerate(model.named_parameters()):

                # sparse net's gradients
                if m.startswith(tuple(['patch_embed', 'cls_token', 'norm', 'pos_embed'])) and False:
                    if params.grad is not None:
                        params.grad *= 0.0

                else:
                    # args.g_prompt_layer_idx : [0, 1]
                    # args.e_prompt_layer_idx : [2, 3, 4]
                    if 'blocks' in m :
                        if 'attn' in m or 'mlp.fc1' in m or 'mlp.fc2' in m:
                            #if 'zeros_weight' in m or 'ones_weight' in m or 'w_m' in m:
                            if 'w_m' in m:
                                if len(m.split('.')) == 5:
                                    # blocks.2.attn.qkv.w_m
                                    # blocks.2.attn.proj.w_m
                                    # blocks.2.mlp.fc1.w_m
                                    blocks, layer, module, submodule, param = m.split('.')
                                    block_layer = getattr(model, blocks)[int(layer)]
                                    ones_weight = getattr(getattr(block_layer, module), submodule).ones_weight
                                else:
                                    import ipdb; ipdb.set_trace()

                                grad_perturb(params.abs(), ones_weight, task_id, args)

                                if params.grad is not None:
                                    grad_list.append(m)                          

                    else:
                        if 'g_prompt' in m or 'e_prompt' in m:
                            if params.grad is not None:
                                grad_list.append(m)
                
                if task_id != 0 and not args.no_pgp:
                    if m == "g_prompt":
                        lz0 = params.grad.data[0][0].size(0)
                        sz0 = params.grad.data[0][0].size()
                        params.grad.data[0][0] = params.grad.data[0][0] - torch.matmul(params.grad.data[0][0].view(lz0, -1),
                                                                                        feature_prefix_gt[0]['key']).view(sz0)
                        lz1 = params.grad.data[1][0].size(0)
                        sz1 = params.grad.data[1][0].size()
                        params.grad.data[1][0] = params.grad.data[1][0] - torch.matmul(params.grad.data[1][0].view(lz1, -1),
                                                                                        feature_prefix_gt[1]['key']).view(sz1)
                    if m == "e_prompt.prompt":
                        params.grad.data[0][0][prompt_id] = params.grad.data[0][0][prompt_id] - torch.matmul(
                            params.grad.data[0][0][prompt_id].view(1, args.length, 768), feature_prefix_et[2]['key']).view(1, args.length, 12, 64)
                        params.grad.data[1][0][prompt_id] = params.grad.data[1][0][prompt_id] - torch.matmul(
                            params.grad.data[1][0][prompt_id].view(1, args.length, 768), feature_prefix_et[3]['key']).view(1, args.length, 12, 64)
                        params.grad.data[2][0][prompt_id] = params.grad.data[2][0][prompt_id] - torch.matmul(
                            params.grad.data[2][0][prompt_id].view(1, args.length, 768), feature_prefix_et[4]['key']).view(1, args.length, 12, 64)

        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

        result_dict = {
            'train/lr': optimizer.param_groups[0]["lr"], 
            'train/Loss': loss.item(),
            'train/Acc@1': acc1.item(), 
            'train/Acc@5': acc5.item()}
        wandb.log(result_dict)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def oneshot_task_inference(model, batch, num_tasks, device, cls_features):

    # Initialize alphas to uniform
    alphas = torch.ones(num_tasks, 1, 1).to(device) / num_tasks
    alphas.requires_grad_(True)

    model.apply(lambda m: setattr(m, "alphas", alphas))
    model.apply(lambda m: setattr(m, "start_task", 0))
    model.apply(lambda m: setattr(m, "num_tasks_learned", num_tasks))

    # Entropy of logits
    shot = 7
    output = model(batch[:shot], task_id = -1, cls_features=cls_features[:shot], mode='infer')
    logits, idx = output['logits'].topk(output['logits'].size(1) // num_tasks, dim=1)
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(1).mean()

    # Gradient wrt alphas
    g, = autograd.grad(entropy, alphas)
    inferred_task = (-g).squeeze().argmax()

    return inferred_task.item()


#@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, cur_id=-1, class_mask=None, args=None, num_tasks=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    #with torch.no_grad():
    output_dict = dict()
    b_cnt = 0
    log_attn = False
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if original_model is not None:
            output = original_model(input)
            cls_features = output['pre_logits']
        else:
            cls_features = None

        if args.infer_type == 'grad':
            infer_id = oneshot_task_inference(model, input, num_tasks, device, cls_features=cls_features)
            if infer_id != task_id:
                print('Grad, Task_id:', task_id, 'Infer_id:', infer_id)
        else:
            infer_id = -1

        with torch.no_grad():
            output = model(input, cls_features=cls_features, infer_id=infer_id, gt_id=task_id)
        logits = output['logits']

        # --- Task-ID inference accuracy (CIL 분석용) ---
        # Prompt ID: 모델 내부 key-matching 결과(prompt_idx)가 실제 task와 일치하는 비율
        # Gradient ID: oneshot_task_inference의 배치 단위 추론 결과
        if args.infer_type == 'grad':
            metric_logger.meters['TaskID_Acc'].update(float(infer_id == task_id) * 100., n=input.shape[0])
        elif 'prompt_idx' in output and output['prompt_idx'] is not None:
            pidx = output['prompt_idx']
            tid_hits = (pidx[:, 0] == task_id).float().mean().item() * 100.
            metric_logger.meters['TaskID_Acc'].update(tid_hits, n=input.shape[0])

        if log_attn:
            output_dict[b_cnt] = {}
            output_dict[b_cnt]['l1_attn'] = model.attn_map[0]
            output_dict[b_cnt]['l2_attn'] = model.attn_map[1]
            output_dict[b_cnt]['l10_attn'] = model.attn_map[9]
            output_dict[b_cnt]['l11_attn'] = model.attn_map[10]
            output_dict[b_cnt]['l12_attn'] = model.attn_map[11]
            output_dict[b_cnt]['logit'] = logits
            output_dict[b_cnt]['target'] = target
            b_cnt += 1

        if args.task_inc and class_mask is not None:
            #adding mask to output logits
            mask = class_mask[task_id]
            mask = torch.tensor(mask, dtype=torch.int64).to(device)
            logits_mask = torch.ones_like(logits, device=device) * float('-inf')
            logits_mask = logits_mask.index_fill(1, mask, 0.0)
            logits = logits + logits_mask

        loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        metric_logger.meters['Loss'].update(loss.item())
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    if 'TaskID_Acc' in metric_logger.meters:
        print('* Task-ID inference Acc {tid.global_avg:.3f} ({itype})'.format(
            tid=metric_logger.meters['TaskID_Acc'], itype=args.infer_type))

    log_stats = {f'train_task': str(cur_id), f'test_task': str(task_id), f'acc': str(metric_logger.meters['Acc@1'])}
    if 'TaskID_Acc' in metric_logger.meters:
        log_stats['task_id_acc'] = '{:.4f}'.format(metric_logger.meters['TaskID_Acc'].global_avg)

    if args.output_dir and utils.is_main_process():
        with open(os.path.join(args.output_dir, 'test_stats.txt'), 'a') as f:
            f.write(json.dumps(log_stats) + '\n')

        if log_attn:
            output_path = os.path.join(args.output_dir, f'task{task_id}_output')
            torch.save(output_dict, output_path)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# @torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    task_id_accs = []

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                            device=device, task_id=i, cur_id=task_id, class_mask=class_mask, args=args, num_tasks=task_id+1)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

        if 'TaskID_Acc' in test_stats:
            task_id_accs.append(test_stats['TaskID_Acc'])

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])

    result_dict = {
        'eval/Avg_acc_till_task': task_id+1,
        'eval/Acc@1': avg_stat[0],
        'eval/Acc@5': avg_stat[1],
        'eval/Loss': avg_stat[2],
    }

    if task_id_accs:
        avg_tid = float(np.mean(task_id_accs))
        result_str += "\tTaskID_Acc({}): {:.4f}".format(args.infer_type, avg_tid)
        result_dict['eval/TaskID_Acc'] = avg_tid

    wandb.log(result_dict)
    
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)

        result_dict = {
            'eval/Forget': forgetting,
            'eval/Backward': backward
        }
        wandb.log(result_dict)
        
    print(result_str)

    return test_stats

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args = None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    model.apply(lambda m: setattr(m, "num_tasks", args.size))
    model.apply(lambda m: setattr(m, "batchwise_prompt", args.batchwise_prompt))

    # Feature for Prefix Prompt
    feature_prefix_g = {}
    feature_prefix_e = {}
    feature_prefix_gt = None
    feature_prefix_et = None

    total_train_time = 0
    total_eval_time = 0

    for task_id in range(args.num_tasks):
        # Transfer previous learned prompt params to the new prompt
        if args.prompt_pool and args.shared_prompt_pool:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                if (prev_end > args.size) or (cur_end > args.size):
                    pass
                else:
                    cur_idx = (slice(None), slice(None), slice(cur_start, cur_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(cur_start, cur_end))
                    prev_idx = (slice(None), slice(None), slice(prev_start, prev_end)) if args.use_prefix_tune_for_e_prompt else (slice(None), slice(prev_start, prev_end))

                    with torch.no_grad():
                        if args.distributed:
                            model.module.e_prompt.prompt.grad.zero_()
                            model.module.e_prompt.prompt[cur_idx] = model.module.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.module.parameters()
                        else:
                            model.e_prompt.prompt.grad.zero_()
                            model.e_prompt.prompt[cur_idx] = model.e_prompt.prompt[prev_idx]
                            optimizer.param_groups[0]['params'] = model.parameters()
                    
        # Transfer previous learned prompt param keys to the new prompt
        if args.prompt_pool and args.shared_prompt_key:
            if task_id > 0:
                prev_start = (task_id - 1) * args.top_k
                prev_end = task_id * args.top_k

                cur_start = prev_end
                cur_end = (task_id + 1) * args.top_k

                with torch.no_grad():
                    if args.distributed:
                        model.module.e_prompt.prompt_key.grad.zero_()
                        model.module.e_prompt.prompt_key[cur_idx] = model.module.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.module.parameters()
                    else:
                        model.e_prompt.prompt_key.grad.zero_()
                        model.e_prompt.prompt_key[cur_idx] = model.e_prompt.prompt_key[prev_idx]
                        optimizer.param_groups[0]['params'] = model.parameters()
     
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        start_time = time.time()
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(model=model, original_model=original_model, feature_prefix_gt=feature_prefix_gt,
                                          feature_prefix_et=feature_prefix_et, criterion=criterion,
                                          data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            wandb.log({'train/epoch': epoch, 'train/task_id': task_id + 1})

            if lr_scheduler:
                lr_scheduler.step(epoch)

        end_time = time.time()
        training_time = end_time - start_time
        total_train_time += training_time

        if not args.no_pgp:
            prefix_rep_g, prefix_rep_e = memory.get_prefix_matrix(data_loader[task_id]['mem'], model, device)

            threshold = 0.50
            feature_prefix_g = memory.update_memory_prefix(prefix_rep_g, threshold, feature_prefix_g)
            feature_prefix_e = memory.update_memory_prefix(prefix_rep_e, threshold, feature_prefix_e)

            feature_prefix_gt = {0: {}, 1: {}}
            feature_prefix_et = {2: {}, 3: {}, 4: {}}
            for layer in feature_prefix_g:
                for item in feature_prefix_g[layer]:
                    temp_feature = feature_prefix_g[layer][item].reshape(feature_prefix_g[layer][item].shape[0], -1)
                    Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose())).to(device)
                    print('g', layer, item, Uf.size())
                    feature_prefix_gt[layer][item] = Uf
            for layer in feature_prefix_e:
                for item in feature_prefix_e[layer]:
                    temp_feature = feature_prefix_e[layer][item].reshape(feature_prefix_e[layer][item].shape[0], -1)
                    Uf = torch.Tensor(np.dot(temp_feature, temp_feature.transpose())).to(device)
                    print('e', layer, item, Uf.size())
                    feature_prefix_et[layer][item] = Uf
                    print("item", item)

        start_time = time.time()
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        
        end_time = time.time()
        total_eval_time = end_time - start_time
        wandb.log({'train/tr_time_sec': total_train_time, 'train/eval_time_sec': total_eval_time}) 

        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')