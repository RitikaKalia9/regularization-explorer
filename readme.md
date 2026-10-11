# Regularization Explorer


An interactive Streamlit app that shows how regularization fights
**overfitting**. Pick a dataset (or upload your own CSV), train a small
neural network with any mix of L2, L1, Dropout, DropConnect, BatchNorm and
early stopping, and see — with annotated charts — what each setting does to
the gap between training and test accuracy.

> **Overfitting** = the model scores well on data it has seen but poorly on
> data it hasn't. **Regularization** = techniques that stop the model from
> memorising the training data.

---

## Features

### Data
- **Dataset dropdown:** FashionMNIST, MNIST, CIFAR-10 (downloaded
  automatically into `./data` on first use).
- **Upload your own CSV:** choose the target column and the share of rows held
  out for testing.
  - Numeric columns are standardised (using training rows only, so there is no
    leakage from the test set).
  - Text columns are one-hot encoded; text columns with more than 30 distinct
    values (e.g. IDs, names) are skipped with a note.
  - The target must have at most 20 distinct classes (classification only).
    Missing feature values are filled with the column median.
  - Files above 50,000 rows are randomly down-sampled.
- **Training-set size** (image datasets): 500 / 2,000 / 10,000 — fewer
  examples means more overfitting.

### Regularization and training controls
| Control | What it does | What to expect |
|---|---|---|
| L2 weight decay | Penalises large weights | Smaller weights, smaller gap |
| L1 lambda | Pushes many weights to exactly 0 | Spike at 0 in the weight chart |
| Dropout | Randomly switches off neurons during training | Lower train accuracy, smaller gap |
| DropConnect | Randomly switches off individual weights | Like Dropout, finer-grained |
| BatchNorm | Normalises each layer's outputs | Faster training; not always less overfitting |
| Early stopping | Stops when test loss stops improving (patience 3) | Avoids the late-epoch rise in test loss |
| Learning rate, epochs | Standard training settings | — |

The network is one hidden layer of 256 units (`inputs → 256 → classes`), sized
automatically for the chosen dataset (e.g. 784 → 256 → 10 for FashionMNIST).

### Charts and results
- **Metrics + verdict:** train accuracy, test accuracy and the generalization
  gap, with a coloured verdict (overfitting / underfitting / generalizing
  well).
- **Loss chart:** train (blue) vs test (orange), gap shaded, green star at the
  best epoch (where early stopping would stop).
- **Accuracy chart:** the same view in plain percentages.
- **Weights chart:** log-scale histogram plus "% of weights ≈ 0", which makes
  the effect of L1 and L2 visible.
- **Sample predictions** (image datasets): 12 test images, green = correct,
  red = wrong.
- **Compare all runs:** train-vs-test bar chart for every run, an overlay of
  every run's test loss, and a results table with the best run highlighted.
  A "Clear all runs" button resets the session.
- Plain-English captions under each chart and a "New here?" guide in the app.

### Reproducibility
The random seed is reset before every model is built (`set_seed()`), as in
the lab code, so runs are directly comparable.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires Python 3.9+. `requirements.txt` points to the CPU-only PyTorch
wheels, which keeps the install small. On a machine with a GPU the app uses it
automatically.

## How to use

1. Choose a **dataset** in the sidebar (or upload a CSV and pick the target
   column).
2. Press **Train** with all regularization off to get a baseline.
3. Switch on one technique at a time, press **Train** again, and compare.
4. Read the **gap** (train − test accuracy). A large gap means overfitting;
   low train accuracy means underfitting.
5. Use the **Compare all runs** section to see which setting shrinks the gap
   without hurting test accuracy.

Tip: small datasets (a few hundred rows) overfit easily and make the best
demos.

## Deploying (Streamlit Community Cloud)

1. Push `app.py`, `requirements.txt` and this readme to a GitHub repo (do not
   commit `data/`).
2. On share.streamlit.io choose **Create app**, select the repo, branch and
   `app.py`, then **Deploy**.
3. The free tier is CPU-only with limited RAM, so use the smaller training
   sizes and fewer epochs when demoing.

---

## Example results (FashionMNIST, from the lab)

All runs used `train_size = 2000`, `lr = 0.001`, `epochs = 20` unless noted.

| Setting | Train acc | Test acc | Gap | Verdict |
|---|---|---|---|---|
| Baseline (no reg) | 0.914 | 0.813 | 0.101 | Overfitting |
| L2 = 0.001 | 0.909 | 0.813 | 0.096 | Overfitting |
| L2 = 0.01 | 0.843 | 0.792 | 0.051 | Overfitting |
| L1 = 0.0001 | 0.884 | 0.810 | 0.074 | Overfitting |
| Dropout p = 0.2 | 0.894 | 0.806 | 0.089 | Overfitting |
| **Dropout p = 0.5 (best overall)** | **0.896** | **0.820** | **0.076** | Generalizing well |
| Dropout p = 0.8 | 0.841 | 0.801 | 0.039 | Generalizing well |
| BatchNorm | 0.983 | 0.817 | 0.166 | Overfitting |
| weight_decay = 0.5 (+BatchNorm) | 0.681 | 0.670 | 0.011 | Underfitting |
| DropConnect p = 0.35 (+BatchNorm) | 0.957 | 0.795 | 0.162 | Overfitting |
| Early stopping (+BatchNorm, stopped epoch 5) | 0.898 | 0.798 | 0.101 | Overfitting |
| Part F: Dropout 0.5 + L1 0.001 | 0.771 | 0.760 | 0.011 | Generalizing well |
| Part F: Dropout 0.5 + L1 0.001 + BatchNorm | 0.818 | 0.764 | 0.054 | Generalizing well |

**Headline finding:** Dropout p = 0.5 on its own was the best single
configuration. Combining it with L1 (Part F) did not beat it (0.760 and 0.764
vs 0.820 test accuracy) — stacking regularization techniques is not
automatically better than the single best one.

Caveats: the weight_decay = 0.5, DropConnect and early-stopping runs also had
BatchNorm switched on, so they are not a clean single-variable change from the
baseline. Turning on Dropout and DropConnect together triggers a warning,
because the lab compares them one at a time (Part D.4).

---

## Project files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app (data loading, model, training, charts) |
| `requirements.txt` | Python dependencies (CPU-only PyTorch index) |
| `formulas.txt` | Formulas used in the lab |
| `readme.md` | This file |

## Limitations

- Classification only (CSV targets must have ≤ 20 classes); no regression.
- One fixed architecture: a single hidden layer of 256 units.
- The "test" split is also used for early stopping and for the per-epoch
  curves, as in the lab. For a strict evaluation, hold out a third split.
- Results are from a single seed; differences of a point or less between runs
  may be noise.
