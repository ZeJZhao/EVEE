# EVEE

Official implementation of **EVEE: Event-Based Online Adaptation for Matching on Unknown Targets**, accepted at **ECCV 2026**.

EVEE performs event-driven online adaptation for local feature detection and matching on previously unseen targets. It uses event streams as proxy supervision and combines TraW, EVB, and ReWeight for efficient target-aware matching.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/ZeJZhao/EVEE.git
cd EVEE
```

### 2. Create Environment

Create a conda environment, install PyTorch for your CUDA version, then install the packages listed in `requirements.txt`.

```bash
conda create -n evee python=3.10 -y
conda activate evee

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

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
--infer_out_dir ./my_results
--matches_dump_dir ./my_results/matches
```

## Citation

If EVEE is useful for your research, please cite:

```bibtex
  @inproceedings{zhao2026evee,
    title     = {{EVEE}: Event-Based Online Adaptation for Matching on Unknown Targets},
    author    = {Zhao, Zejing and Ju, Cheng and Zhang, Yanwen and Namiki, Akio},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026}
  }
```
