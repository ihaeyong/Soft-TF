# Soft-TransFormers for Continual Learning (Soft-TF)

Official code for **"Soft-TransFormers for Continual Learning"** (Haeyong Kang, Chang D. Yoo).

Soft-TF adapts a **frozen** pre-trained Transformer by learning task-specific
**soft subnetworks** — real-valued multiplicative masks over the query/key/value/
projection weights of selected self-attention layers. The masks are **initialized
at one**, so optimization starts exactly at the pre-trained solution and one SGD
step in mask space equals a `diag(θ⊙²)`-preconditioned step in weight space. This
keeps every task within a bounded drift of the backbone, so forgetting is
structurally eliminated. Soft-TF is a **plug-in**: it composes with prompt-based
learners (L2P, DualPrompt, HiDe-Prompt, NoRGa) and keeps inference cost identical
to the unmodified backbone.

This repository contains the **L2P / DualPrompt + Soft-TF** track. The
HiDe-Prompt / NoRGa + Soft-TF rows are reproduced on top of the
[HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt) / NoRGa codebases (see
[Reproducing Table I](#reproducing-table-i)).

---

## Environment

```bash
conda create -n soft_tf python=3.10 -y
conda activate soft_tf
pip install -r requirements.txt
```

Core dependencies (`requirements.txt`):

```
torch==1.13.1  torchvision==0.14.1  timm==0.6.7
numpy==1.25.0  scikit-learn==1.3.0  matplotlib  wandb
```

`wandb` is only used for logging; the reproduction script sets `WANDB_MODE=offline`
so no account is required. The ViT-B/16 (Sup-21K) backbone is downloaded
automatically by `timm`.

## Data preparation

Pass your dataset root to `--data-path` (default `./datasets/`). If a dataset is
absent, CIFAR-100 / ImageNet-R / CUB-200 are downloaded there on first run.

## Reproducing Table I

Table I reports Vision Class-Incremental Learning on 10/20-Split-CIFAR100 and
10-Split-ImageNet-R (accuracy, forgetting, trainable params, train/test time).
Soft-TF is toggled by `--subnet` and, by default, masks the last three attention
layers **L[10,11,12]** (the E-Prompt layers):

| `--subnet` | Table I row                         |
|:-----------|:------------------------------------|
| *(omit)*   | DualPrompt* (baseline)              |
| `soft`     | **DualPrompt + Soft-TF-L[10,11,12]**|
| `adapter`  | DualPrompt + Adapter                |
| `lora`     | DualPrompt + LoRA                   |

One-line reproduction of the DualPrompt track (GPU id as first arg):

```bash
# Soft-TF on all three datasets
./scripts/reproduce_table1.sh 0

# every variant on 10-Split-CIFAR100
./scripts/reproduce_table1.sh 0 c100_10 all

# a single cell: Soft-TF on 10-Split-ImageNet-R
./scripts/reproduce_table1.sh 0 imr soft
```

Equivalently, a single run by hand:

```bash
python main.py 10cifar100_dualprompt_pgp \
    --model vit_base_patch16_224 --output_dir ./output \
    --epochs 20 --no_pgp --subnet soft        # DualPrompt + Soft-TF, 10-Split-CIFAR100
```

Configs: `10cifar100_dualprompt_pgp`, `20cifar100_dualprompt_pgp`,
`imr_dualprompt_pgp`, `cub200_dualprompt_pgp`, `tinyimagenet_dualprompt_pgp`.
Each run prints **Avg. Acc / Forgetting** and the trainable-parameter count that
fill the corresponding Table I cell.

**HiDe-Prompt / NoRGa + Soft-TF rows.** These build on the HiDe-Prompt / NoRGa
codebases, where Soft-TF is enabled with `--soft_tf_layers 9 10 11`. The exact
commands (shared TII head for a controlled ±Soft-TF comparison) are in
`scripts/reproduce_hide_norga.md`.

## Evaluation

```bash
python main.py 10cifar100_dualprompt_pgp --subnet soft --eval
```

## Acknowledgements

Built on the code of
[PGP](https://github.com/JingyangQiao/prompt-gradient-projection),
[L2P](https://github.com/JH-LEE-KR/l2p-pytorch),
[DualPrompt](https://github.com/JH-LEE-KR/dualprompt-pytorch), and
[HiDe-Prompt](https://github.com/thu-ml/HiDe-Prompt). We thank the authors for
releasing their implementations.

## Citation

```bibtex
@article{kang2026softtf,
  title  = {Soft-TransFormers for Continual Learning},
  author = {Kang, Haeyong and Yoo, Chang D.},
  journal= {arXiv preprint},
  year   = {2026}
}
```

> Under review at *IEEE Transactions on Neural Networks and Learning Systems*.
> Please update the citation once the arXiv id / DOI is available.

## License

Released under the Apache-2.0 License (see `LICENSE`).
