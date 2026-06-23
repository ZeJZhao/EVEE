# EVEE

Official implementation of **EVEE: Event-Based Online Adaptation for Matching on Unknown Targets**, accepted at **ECCV 2026**.

EVEE performs event-driven online adaptation for local feature detection and matching on previously unseen targets. It uses event streams as proxy supervision and combines TraW, EVB, and ReWeight for efficient target-aware matching.

Our method was developed and tested on **Ubuntu 22.04/24.04**. Compatibility with macOS and Windows has **not yet been verified**.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ZeJZhao/EVEE.git
cd EVEE
```

### 2. Create Environment

Create a conda environment, install PyTorch for your CUDA version, then install the base packages and Mamba-related CUDA extensions.

```bash
conda create -n evee python=3.10 -y
conda activate evee

pip install pip==24.0 setuptools==69.5.1 wheel==0.43.0
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-build-isolation -r requirements_M.txt
```

The pinned `setuptools` version is required because newer conda environments may install a very recent `setuptools` that no longer exposes `pkg_resources`, while PyTorch 2.1.2 still imports it during CUDA extension builds.

`cupy-cuda12x` is included for CUDA runs and matches the tested `evee` environment. Without CuPy, RGB tensors may stay on CPU while masks are moved to CUDA, which can trigger a CPU/CUDA device mismatch in the current release code.

`transformers` is pinned to `4.28.1` for compatibility with `mamba-ssm==2.2.4` and `torch==2.1.2`. Newer `transformers` releases may require newer PyTorch versions and break `mamba_ssm` imports.

`causal-conv1d` and `mamba-ssm` are installed with `--no-build-isolation` because their build scripts import `torch` during installation. If they are installed through a normal isolated pip build, pip may report `ModuleNotFoundError: No module named 'torch'` even after PyTorch has already been installed.

### 3. Prepare Weights

Place the pretrained weights in the following paths:

```text
weights/PrEEVEE.pth
core/MagicPoint_Weight/MagicPoint_Weight.pth
```

### 4. Prepare Data

Download the example dataset from Google Drive:

```text
https://drive.google.com/file/d/1zl7JL_-8Qu4u18kL4R_5tw1XFNK4D24i/view?usp=drive_link
```

After downloading, extract it under `dataset/`.

The expected data layout is:

```text
dataset/
└── bunny_racer/
    └── video-05-pixel_7-PXL_20230728_005055979.TS/
        ├── images/
        │   ├── frame_00000.png
        │   └── ...
        ├── events/
            ├── frame_00000.png
            └── ...

```

`images/` contains RGB frames, `events/` contains event-frame supervision, and `masks/` contains foreground masks.

### 5. Run EVEE

```bash
python EVEE.py
```

### 6. Outputs

After running, the main outputs are:

```text
Output_result/
├── 000000.png
├── 000001.png
├── ...
└── matches/
    ├── matches_00000_00001.txt
    ├── ...
    └── online_head_best.pth
```

`FPS_result.txt` is written in the current working directory and stores the final FPS values.

## Common Options

```bash
# Change output directory
--infer_out_dir ./Output_Results
--matches_dump_dir ./Output_Results/matches
```

## Citation

If EVEE is useful for your research, please cite:

```bibtex
@inproceedings{evee2026,
  title     = {EVEE: Event-Based Online Adaptation for Matching on Unknown Targets},
  author    = {Zhao, Zejing and Ju, Cheng and Zhang, Yanwen and Namiki, Akio},
  booktitle = {ECCV},
  year      = {2026}
}
```
