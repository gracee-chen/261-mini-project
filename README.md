# Caltech-101 Image Classification Benchmark

CS 261 mini project — benchmarks 6 models (ResNet-50, EfficientNet-B2, ViT-B/16, ConvNeXt-Tiny, EVA-02-Small, SVM) on Caltech-101.

## Run on Colab

```python
# 1. Clone the repo
!git clone https://github.com/<your-username>/<your-repo>.git
%cd <your-repo>

# 2. Install dependencies
!pip install -r requirements.txt

# 3. Download Caltech-101 dataset
!kaggle datasets download -d imbikramsaha/caltech-101 -p data/ --unzip

# 4. Train all models
%cd model
!python run_all.py

# 5. Train a single model
!python train.py --model eva02_small

# 6. Run ablation studies
!python experiment/run_ablations.py

# 7. Generate paper figures
!python generate_paper_figures.py
```

## Project Structure

```
model/              Training, evaluation, and model code
model/experiment/   Ablation study scripts
latex/              Paper source files (.tex)
fig/                Generated figures
data/               Dataset
```
