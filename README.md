# 🏃 MotionGPT3 — Team 3
### Human Motion as a Foreign Language (NeurIPS 2023)

> **Google Colab Demo** · Runs on free T4 GPU · Gradio web UI · 4 motion tasks

---

## 📋 Table of Contents
- [Overview](#overview)
- [Tasks](#tasks)
- [Prerequisites](#prerequisites)
- [How to Run](#how-to-run)
- [Gradio UI — 24-Hour Note](#gradio-ui--24-hour-note)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Credits](#credits)

---

## Overview

MotionGPT treats **human body motion as a foreign language**. A VQ-VAE tokenizer encodes 3D motion sequences into discrete tokens, and a **Flan-T5** language model is fine-tuned to reason over both text and motion tokens in a unified way.

This demo runs the pretrained `MotionGPT-base` checkpoint on the **HumanML3D** dataset and exposes all four tasks through:
- Individual notebook cells (for step-by-step control)
- An interactive **Gradio web UI** with a public shareable link

---

## Tasks

| Task | Description | Input | Output |
|------|-------------|-------|--------|
| 📝 **Text → Motion** | Generate 3D skeleton animation from a text description | Text prompt | Animated GIF |
| 🎬 **Motion → Text** | Caption a motion sequence in natural language | `.npy` file (T×263) | Text caption |
| 🔵 **In-Between** | Fill in the masked middle 50% of a motion | `.npy` file | Completed motion GIF |
| 🟠 **Prediction** | Predict future frames from the first 20% of a motion | `.npy` file | Predicted future GIF |

---

## Prerequisites

### Hardware
- **GPU required**: T4 GPU (free on Google Colab)
- Go to `Runtime → Change runtime type → T4 GPU` **before** running any cell

### Software (auto-installed by the notebook)
| Package | Version | Purpose |
|---------|---------|---------|
| pytorch-lightning | 2.2.5 | Training framework |
| transformers | >=4.36.0,<4.42.0 | Flan-T5 backbone |
| gradio | >=3.50.0,<4.0.0 | Web UI |
| moviepy | 1.0.3 | GIF rendering |
| omegaconf | latest | Config management |
| gdown | latest | Google Drive downloads |
| scipy, einops, rich, tqdm | latest | Utilities |
| bert-score, spacy | latest | Evaluation metrics |

### External Downloads (auto-downloaded by notebook)
| Asset | Size | Source |
|-------|------|--------|
| MotionGPT-base checkpoint | ~500 MB | HuggingFace (`OpenMotionLab/MotionGPT-base`) |
| SMPL body model | ~50 MB | Google Drive |
| Flan-T5-base | ~1 GB | HuggingFace (`google/flan-t5-base`) |
| T2M evaluators (mean/std) | ~200 MB | Google Drive |

---

## How to Run

### Option A — Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `team3_motiongpt3.py` **or** open the original notebook from the Colab link in the file header
3. Set runtime: `Runtime → Change runtime type → T4 GPU`
4. Run all cells: `Ctrl+F9` (or `Runtime → Run all`)
5. **After Step 1 installs complete** → restart runtime when prompted → run all cells again (`Ctrl+F9`)
6. Wait for model to load (~2–3 min)
7. Use the Gradio link that appears at the bottom of Step 5

### Step-by-Step Order

| Step | What It Does |
|------|-------------|
| Step 1 | Install packages (then **restart runtime**) |
| Step 2 | Download checkpoint, SMPL, Flan-T5, T2M evaluators |
| Step 3 | Load and initialize the MotionGPT model |
| Step 4 | Define utility functions (`render_gif`, `show_gif`, etc.) |
| Task 1 | Text → Motion |
| Task 2 | Motion → Text |
| Task 3 | In-Between |
| Task 4 | Prediction |
| Step 5 | Launch Gradio interactive UI |

---

## Gradio UI — 24-Hour Note

> ⚠️ **The Gradio public URL expires after 24 hours.**

This is a Gradio/ngrok limitation — the tunnel closes after 24 hours. To get a fresh URL:

1. Open the notebook in Colab
2. Press **Ctrl+F9** to re-run all cells
3. After pip installs finish → **Restart runtime** → **Ctrl+F9** again
4. A new public URL will appear when Step 5 runs

**What gets reset when Colab restarts:**
- All downloaded files (checkpoint, SMPL, deps) — re-downloaded automatically
- All pip packages — re-installed automatically
- Generated GIFs — regenerate by running task cells

**Tip:** To save results permanently, mount Google Drive and save to `/content/drive/MyDrive/`.

---

## File Structure

```
MotionGPT3_repo/
├── team3_motiongpt3.py   # Main notebook converted to .py (all steps + tasks)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

When running on Colab, the notebook auto-creates:
```
/content/MotionGPT/           # Cloned from OpenMotionLab/MotionGPT
├── checkpoints/              # Downloaded checkpoint
├── configs/                  # Model configs
├── mGPT/                     # Model source code
│   ├── config.py             # Patched for eval resolver
│   └── models/mgpt.py        # Patched for prediction task lengths
├── deps/
│   ├── smpl/                 # SMPL body model
│   ├── flan-t5-base/         # Language model
│   ├── t2m/                  # HumanML3D evaluators
│   └── glove/                # GloVe word vectors
└── results/                  # Generated GIFs saved here
```

---

## Dependencies

Install manually (if not using Colab auto-install):

```bash
pip install pytorch-lightning==2.2.5
pip install "transformers>=4.36.0,<4.42.0"
pip install "tokenizers>=0.15.0"
pip install omegaconf einops rich tqdm scipy
pip install moviepy==1.0.3 imageio imageio-ffmpeg matplotlib
pip install "gradio>=3.50.0,<4.0.0"
pip install gdown bert-score spacy
python -m spacy download en_core_web_sm
```

> **Note:** PyTorch is assumed to already be installed (Colab provides 2.2+). For local setup, install torch first: https://pytorch.org/get-started/locally/

---

## Architecture

```
Text Prompt
     │
     ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  T5 Text    │────▶│  Flan-T5-base LM │────▶│  Motion Token   │
│  Tokenizer  │     │  (250M params)   │     │  Decoder        │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                    ┌──────────────────┐               ▼
                    │  VQ-VAE Decoder  │◀──── Motion Tokens
                    │  (codebook=512)  │       <motion_id_N>
                    └────────┬─────────┘
                             │
                             ▼
                    3D Joint Positions (T × 22 × 3)
                             │
                             ▼
                    Animated GIF (matplotlib)
```

---

## Patches Applied

The notebook applies two source-level patches to the upstream MotionGPT repo:

1. **`mGPT/config.py`** — adds `replace=True` to `OmegaConf.register_new_resolver("eval", eval)` to allow re-registration on kernel restarts without errors.

2. **`mGPT/models/mgpt.py`** — adds `lengths.append(motion.shape[1])` inside the `pred` task branch (was missing in the original, causing empty `length` output for Task 4).

---

## Credits

- **Original paper:** [MotionGPT: Human Motion as a Foreign Language](https://arxiv.org/abs/2306.14795) (NeurIPS 2023)
- **Original repo:** [OpenMotionLab/MotionGPT](https://github.com/OpenMotionLab/MotionGPT)
- **Dataset:** [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- **Demo implementation:** Team 3 — University of North Carolina Charlotte

---

## License

This demo code is for educational purposes. The MotionGPT model and weights are subject to the [original license](https://github.com/OpenMotionLab/MotionGPT/blob/main/LICENSE).
