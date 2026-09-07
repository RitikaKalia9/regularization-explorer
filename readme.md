# Regularization Explorer: MINI PROJECT LAB 4

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

## Results

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

**Headline finding:** Dropout p = 0.5 on its own was the best-performing
single configuration across every run. Combining it with L1 (Part F) did
not beat it — both combined models scored lower test accuracy (0.760 and
0.764 vs 0.820) — confirming the lab's own warning that stacking
regularization techniques isn't automatically better than the single best
one. Full analysis, hand calculations (Table 2), and conclusions
(Table 3) are in `Regularization_Explorer_Report.docx`.

## Notes for your report / record sheet

- The seed is reset before every model build (`set_seed()`), same as the
  lab code, so runs are comparable.
- The `weight_decay = 0.5`, DropConnect, and early-stopping runs above
  also had BatchNorm switched on, so they aren't a clean single-variable
  change from baseline — see the report for the full caveat.
- If you turn on both Dropout and DropConnect at once the app warns you —
  the lab asks you to compare them one at a time (Part D.4), not combine
  them.
- Near-zero weight fraction is logged per run in the results table, which
  is what you need for the L1 task in Part B.4.

## Status

- [x] App (all required + extra-credit features)
- [x] Screenshots (13, one per run above)
- [x] Report (`Regularization_Explorer_Report.docx`) — results, analysis,
      Table 1/2/3, Part F, all screenshots embedded
- [ ] Reread Table 3 in the report and adjust it to sound like your own
      conclusions before submitting
- [ ] Add your roll number to the report header
