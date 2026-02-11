# DL Healthcare Assistant – Fast Track Starter

Implements all Parts 1–5 from the assignment:
- Part 1: Text classification (LSTM + Encoder-only Transformer) with dropout/L2 grid + early stopping + gradient norms
- Part 2: Medical image classification (ResNet18 vs ResNet50) + learning curves + prediction visualizations
- Part 3: VAE image generation + recon/KL curves + samples
- Part 4: Deployment optimization (pruning + dynamic quantization) + size/speed/accuracy benchmark
- Part 5: Ethics template + optional subgroup audit by simple proxies

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Part 1
python -m src.text.run_grid --model lstm --epochs 12
python -m src.text.run_grid --model transformer --epochs 12

# Part 2
python -m src.vision.train_vision --arch resnet18 --epochs 10
python -m src.vision.train_vision --arch resnet50 --epochs 10
python -m src.vision.visualize_predictions --arch resnet18 --n 10

# Part 3
python -m src.generative.train_vae --epochs 20

# Part 4
python -m src.deploy.optimize_text_model --runs_dir runs/text --pick best

# Part 5
See report_templates/ethics_report.txt
```

Outputs are written under `runs/` (plots, tables, saved models).
