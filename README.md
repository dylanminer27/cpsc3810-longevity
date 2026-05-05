# Do Lifestyle and Work Habits Predict Longevity?

This project builds linear regression, multinomial logistic regression, and K-Means
clustering models from scratch to predict and analyze longevity from lifestyle
features in the *Work-Life Balance and Longevity Dataset*.

## Setup

```bash
# 1. Clone this repository:
git clone https://github.com/dylanminer27/cpsc3810-longevity.git
cd cpsc3810-longevity

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the dataset file at data/raw/quality_of_life.csv
#    (download from https://www.kaggle.com/datasets/oluwatosinadewale/quality-of-life-data)
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
