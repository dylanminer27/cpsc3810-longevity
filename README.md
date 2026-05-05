# Do Lifestyle and Work Habits Predict Longevity?

This project builds linear regression, multinomial logistic regression, and K-Means
clustering models from scratch to predict and analyze longevity from lifestyle
features in the *Work-Life Balance and Longevity Dataset*.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

## Reproduce all results

```bash
python -m src.train
```

Running `src/train.py` will:

1. Load `data/raw/quality_of_life.csv`, clean and encode it, and save train / val / test splits to `data/processed/`.
2. Fit the scikit-learn baselines (`LinearRegression`, `LogisticRegression`, `KMeans`) and print results.
3. Fit the custom `LinearRegression` (normal equation, pseudoinverse, and gradient-descent variants) and the custom `LogisticRegression` (multinomial softmax). 
4. Write numbers to `results/metrics.json`.
5. Render figures to `results/figures/`.


## Repository layout

```
.
├── data/
│   ├── raw/                    # quality_of_life.csv 
│   └── processed/              # train/val/test splits 
├── src/
│   ├── data.py                 # load, clean, encode, split, standardize
│   ├── optimizer.py            # Optimizer: GD / momentum GD / SGD / momentum SGD
│   ├── model.py                # LinearRegression, LogisticRegression
│   ├── evaluate.py             # MSE, R^2, accuracy, precision/recall/F1
│   ├── plots.py                # loss curves, confusion matrix, clusters
│   └── train.py                
├── results/
│   ├── metrics.json            # final numbers 
│   └── figures/                # figures
└── report/
    └── final_report.pdf        # final written report
```
