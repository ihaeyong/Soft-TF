# HiDe-Prompt / NoRGa + Soft-TF rows of Table I

These rows are produced on top of the **HiDe-Prompt / NoRGa** codebase, into which
Soft-TF is ported. Clone that codebase alongside this repo and run its `main.py`
with the same ViT-B/16 backbone. Soft-TF is enabled by masking the last three
attention layers with `--soft_tf_layers 9 10 11` (0-indexed → L[10,11,12]).

Protocol used in the paper: `--shuffle False`, seeds `59` (10-split), `42`
(20-split), `43` (ImageNet-R). Each method first trains a Task-Inference (TII)
head **once**, shared by the `base` and `soft` runs, so the ±Soft-TF comparison is
controlled — the accuracy gain comes purely from representation specialization,
not from a better task selector.

## 20-Split-CIFAR100 (seed 42)

```bash
# 1) TII head (shared by base and +Soft-TF)
python main.py cifar100_hideprompt_5e \
    --original_model vit_base_patch16_224 --model vit_base_patch16_224 \
    --batch-size 128 --data-path ./local_datasets/ \
    --output_dir ./output/hide_cifar100_20_tii_seed42 \
    --sched constant --seed 42 --num_tasks 20 --train_inference_task_only \
    --ca_storage_efficient_method covariance --lr 0.0005 --ca_lr 0.005 \
    --crct_epochs 30 --epochs 20 --shuffle False

# 2) main run — set VARIANT=base (drop the flag) or +Soft-TF (keep it)
python main.py cifar100_hideprompt_5e \
    --model vit_base_patch16_224 --original_model vit_base_patch16_224 \
    --batch-size 128 --epochs 50 --data-path ./local_datasets/ \
    --ca_lr 0.05 --seed 42 --num_tasks 20 --size 20 --prompt_momentum 0.01 \
    --reg 0.1 --length 5 --sched step --larger_prompt_lr \
    --ca_storage_efficient_method covariance \
    --trained_original_model ./output/hide_cifar100_20_tii_seed42 \
    --output_dir ./output/hide_cifar100_20_soft_seed42 \
    --soft_tf_layers 9 10 11 --shuffle False --reset
```

## 10-Split-ImageNet-R (seed 43)

Same two-step recipe with config `imr_hideprompt_5e`, seed `43`,
`--ca_lr 0.005 --crct_epochs 30 --length 20 --reg 0.5 --sched cosine`, and
`--soft_tf_layers 9 10 11` for the +Soft-TF run.

## NoRGa

Identical to the above with config `cifar100_norgaprompt` (add `--gate_act sigmoid`).

The runs report **Avg. Acc / Forgetting** and a `TII Acc` line; these fill the
`HiDe-Prompt + Soft-TF` and `NoRGa + Soft-TF` cells of Table I.
