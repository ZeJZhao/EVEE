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

Install PyTorch for your CUDA version first, then install the remaining packages:

```bash
conda create -n evee python=3.10 -y
conda activate evee

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow mamba-ssm
```

CuPy is optional. If it is installed and compatible with your CUDA version, EVEE will use it automatically.

### 3. Prepare Weights

Place the pretrained weights in the following paths:

```text
weights/PrEEVEE.pth
weights/online_head_best.pth
core/MagicPoint_Weight/MagicPoint_Weight.pth
```

For ReWeight, set the checkpoint path explicitly:

```bash
export EVEE_REWEIGHT_CKPT=./weights/PrEEVEE.pth
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
python EVEE.py \
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


## Notes

- Run commands from the repository root so relative paths resolve correctly.
- `events/` is used for online proxy supervision. If no valid event frames are found, the training stage is skipped.
- Visualization frames are saved with the original RGB background while masks are still used for model-side ROI gating.
- The default output directory is configured in `utils/reader/reader.py`.

## Citation

If EVEE is useful for your research, please cite:

```bibtex
@inproceedings{evee2026,
  title     = {EVEE: Event-Based Online Adaptation for Matching on Unknown Targets},
  booktitle = {ECCV},
  year      = {2026}
}
```
