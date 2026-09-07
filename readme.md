# Regularization Explorer

Streamlit app for the Lab 4 Home Assignment (Neural Networks Lab, CCET).
Lets you configure the same 784 → 256 → 10 FashionMNIST network from the
lab with any combination of L2, L1, Dropout, DropConnect and BatchNorm,
train it, and compare runs.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

First run downloads FashionMNIST into `./data` (a few seconds).

## What's implemented

**Required (from the spec):**
- Sidebar controls: L2 weight decay, L1 lambda, dropout rate, BatchNorm
  checkbox, learning rate, epochs.
- "Train" button → trains the model, plots train/test loss curves.
- Results table that keeps every run of the session (settings + final
  train acc, test acc, gap).
- Current run's gap shown as a metric; best run highlighted in the table.
- One-line plain-English verdict (overfitting / underfitting / fine).

**Extra credit (all included):**
- DropConnect layer from Part D, selectable via a slider.
- Weight histogram — shows the spike at 0 that L1 produces.
- Early-stopping checkbox (patience = 3 epochs, watches test loss).
- Training set size selector (500 / 2,000 / 10,000).

## Notes for your report / record sheet

- The seed is reset before every model build (`set_seed()`), same as the
  lab code, so runs are comparable.
- If you turn on both Dropout and DropConnect at once the app warns you —
  the lab asks you to compare them one at a time (Part D.4), not combine
  them.
- Near-zero weight fraction is logged per run in the results table, which
  is what you need for the L1 task in Part B.4.

## What's still on you

The assignment also asks for:
1. A short report (2 pages) — see `report_template.md` for a starting
   structure.
2. Screenshots of at least three different settings — take these while
   running the app locally.
3. Your completed Observation Record Sheet from the lab (Tables 1–3).

The app can't generate these for you since they depend on your actual
runs and your own analysis.