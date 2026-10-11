"""
Regularization Explorer — Lab 4 Home Assignment (Neural Networks Lab, CCET)

A Streamlit app that lets you pick a dataset (FashionMNIST, MNIST, CIFAR-10
or your own CSV), train a small one-hidden-layer network with any mix of
L2 weight decay, L1 penalty, Dropout, DropConnect and BatchNorm, and see
-- with annotated charts -- how each setting changes overfitting.

Run with:
    streamlit run app.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

SEED = 42
TRAIN_COLOR = "#2a78d6"   # blue  = training data
TEST_COLOR = "#e8743b"    # orange = unseen (test) data
GAP_COLOR = "rgba(232,116,59,0.18)"

IMAGE_DATASETS = {
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "MNIST": torchvision.datasets.MNIST,
    "CIFAR-10": torchvision.datasets.CIFAR10,
}
CSV_NAME = "Upload my own CSV"


def set_seed(seed=SEED):
    """Reset the seed before every model so runs are comparable (Q: 'why
    did we reset the seed before building every model?')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading dataset (first time only)...")
def load_image_dataset(name):
    cls = IMAGE_DATASETS[name]
    transform = transforms.ToTensor()
    train_full = cls(root="./data", train=True, download=True, transform=transform)
    test_full = cls(root="./data", train=False, download=True, transform=transform)
    sample, _ = train_full[0]
    return {
        "train": train_full,
        "test": test_full,
        "in_dim": int(np.prod(sample.shape)),
        "n_classes": len(train_full.classes),
        "class_names": [str(c).split(" - ")[-1] for c in train_full.classes],
        "is_image": True,
    }


def make_train_subset(train_full, train_size):
    set_seed()
    indices = torch.randperm(len(train_full))[:train_size]
    return Subset(train_full, indices)


def prepare_csv(df, target, test_frac, max_rows):
    """Turn an uploaded table into train/test tensors.

    Features: numeric columns are standardised, text columns are one-hot
    encoded (skipped if they have more than 30 distinct values). Target:
    any column with a modest number of distinct values, label-encoded.
    Standardisation statistics come from the TRAIN split only, so nothing
    leaks from the test set.
    """
    notes = []
    df = df.dropna(subset=[target]).copy()
    y_raw = df[target]
    n_classes = y_raw.nunique()
    if n_classes < 2:
        raise ValueError("The target column has only one distinct value.")
    if n_classes > 20:
        raise ValueError(
            f"The target column has {n_classes} distinct values, so it looks "
            "like a number to predict (regression). This app does "
            "classification -- pick a column with at most 20 classes."
        )
    classes = sorted(y_raw.astype(str).unique())
    y = y_raw.astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()

    X = df.drop(columns=[target])
    keep = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            keep.append(col)
        elif X[col].nunique() <= 30:
            keep.append(col)
        else:
            notes.append(f"Skipped '{col}' (text column with >30 distinct values).")
    if not keep:
        raise ValueError("No usable feature columns were found.")
    X = pd.get_dummies(X[keep], dtype=float)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    X = X.to_numpy(dtype=np.float32)

    if len(X) > max_rows:
        notes.append(f"Using a random sample of {max_rows:,} of {len(X):,} rows.")
        set_seed()
        pick = torch.randperm(len(X))[:max_rows].numpy()
        X, y = X[pick], y[pick]

    set_seed()
    perm = torch.randperm(len(X)).numpy()
    n_test = max(1, int(round(len(X) * test_frac)))
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    if len(train_idx) < 20:
        raise ValueError("Too few rows left for training -- need at least 20.")

    mean = X[train_idx].mean(0)
    std = X[train_idx].std(0)
    std[std < 1e-8] = 1.0
    X = (X - mean) / std

    train = TensorDataset(torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx]))
    test = TensorDataset(torch.from_numpy(X[test_idx]), torch.from_numpy(y[test_idx]))
    return {
        "train": train, "test": test, "in_dim": X.shape[1],
        "n_classes": len(classes), "class_names": classes, "is_image": False,
        "notes": notes,
    }


# ----------------------------------------------------------------------
# Model — DropConnect layer + the Lab 4 network with optional
# Dropout / BatchNorm / DropConnect
# ----------------------------------------------------------------------
class DropConnectLinear(nn.Module):
    """Dropout's cousin: masks individual weights instead of whole
    neurons. Mask shape = weight matrix shape (hidden x inputs), not the
    neuron list (hidden), which is what makes this DropConnect rather
    than Dropout."""

    def __init__(self, in_features, out_features, p=0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            mask = (torch.rand_like(self.linear.weight) > self.p).float()
            weight = self.linear.weight * mask / (1 - self.p)
        else:
            weight = self.linear.weight
        return F.linear(x, weight, self.linear.bias)


class Net(nn.Module):
    def __init__(self, in_dim=784, n_classes=10, dropout_p=0.0,
                 batchnorm=False, dropconnect_p=0.0, hidden=256):
        super().__init__()
        if dropconnect_p > 0:
            self.fc1 = DropConnectLinear(in_dim, hidden, p=dropconnect_p)
        else:
            self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden) if batchnorm else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.fc2(x)


# ----------------------------------------------------------------------
# Train / evaluate
# ----------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
    return correct / total, loss_sum / total


def train_model(model, train_loader, test_loader, epochs, lr, weight_decay,
                l1_lambda, device, early_stop, progress_cb=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}
    best_test_loss = float("inf")
    patience, bad_epochs = 3, 0

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            if x.size(0) < 2:      # BatchNorm cannot handle a batch of one
                continue
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if l1_lambda > 0:
                l1_norm = sum(p.abs().sum() for p in model.parameters())
                loss = loss + l1_lambda * l1_norm
            loss.backward()
            optimizer.step()

        train_acc, train_loss = evaluate(model, train_loader, device)
        test_acc, test_loss = evaluate(model, test_loader, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if progress_cb:
            progress_cb(epoch, epochs)

        if early_stop:
            if test_loss < best_test_loss - 1e-4:
                best_test_loss = test_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    return history


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def _style(fig, title, ytitle, height=380):
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title="Epoch (one full pass over the training data)",
        yaxis_title=ytitle,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=80, b=10),
        font=dict(size=14),
    )
    return fig


def curve_chart(history, kind):
    """kind = 'loss' or 'acc'. Train vs test curve with the gap shaded."""
    tr, te = history[f"train_{kind}"], history[f"test_{kind}"]
    epochs = list(range(1, len(tr) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=tr, name="Train (seen data)",
                             line=dict(color=TRAIN_COLOR, width=3)))
    fig.add_trace(go.Scatter(x=epochs, y=te, name="Test (unseen data)",
                             line=dict(color=TEST_COLOR, width=3),
                             fill="tonexty", fillcolor=GAP_COLOR))
    if kind == "loss":
        best = int(np.argmin(te))
        fig.add_trace(go.Scatter(
            x=[best + 1], y=[te[best]], mode="markers+text", name="Best test loss",
            marker=dict(size=13, color="green", symbol="star"),
            text=[f"best: epoch {best + 1}"], textposition="top center",
            hoverinfo="skip"))
        title = "Loss per epoch — lower is better"
        ytitle = "Loss (how wrong the predictions are)"
    else:
        title = "Accuracy per epoch — higher is better"
        ytitle = "Accuracy (fraction correct)"
        fig.update_yaxes(tickformat=".0%")
    fig.add_annotation(
        x=epochs[-1], y=(tr[-1] + te[-1]) / 2, xref="x", yref="y",
        text="<b>gap</b>", showarrow=False, xanchor="left", xshift=6,
        font=dict(color=TEST_COLOR, size=13))
    return _style(fig, title, ytitle)


def weight_chart(weights, l1, l2):
    near = float((np.abs(weights) < 1e-3).mean())
    fig = go.Figure(go.Histogram(x=weights, nbinsx=120,
                                 marker=dict(color="#6b5bd2")))
    fig.update_yaxes(type="log")
    reg = []
    if l1 > 0:
        reg.append(f"L1={l1:g}")
    if l2 > 0:
        reg.append(f"L2={l2:g}")
    fig.update_layout(
        title=dict(text="Weight sizes after training"
                         + (f" ({', '.join(reg)})" if reg else " (no L1/L2)"),
                   font=dict(size=18)),
        xaxis_title="Weight value", yaxis_title="Number of weights (log scale)",
        height=360, margin=dict(l=10, r=10, t=60, b=10), font=dict(size=14))
    fig.add_annotation(
        x=0, y=1, xref="x", yref="paper", yanchor="bottom",
        text=f"{near:.1%} of weights are ≈ 0", showarrow=False,
        font=dict(color="#6b5bd2", size=13))
    return fig


def compare_chart(results):
    labels = [f"#{r['run']}" for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["train_acc"] for r in results],
                         name="Train accuracy", marker_color=TRAIN_COLOR))
    fig.add_trace(go.Bar(x=labels, y=[r["test_acc"] for r in results],
                         name="Test accuracy", marker_color=TEST_COLOR))
    fig.update_layout(
        barmode="group", height=380, font=dict(size=14),
        title=dict(text="All runs: the closer the two bars, the less overfitting",
                   font=dict(size=18)),
        xaxis_title="Run number (see table below for settings)",
        yaxis_title="Accuracy", yaxis_tickformat=".0%",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=80, b=10))
    lo = min(min(r["train_acc"], r["test_acc"]) for r in results)
    fig.update_yaxes(range=[max(0, lo - 0.1), 1.0])
    return fig


def overlay_chart(histories, results):
    fig = go.Figure()
    for h, r in zip(histories, results):
        fig.add_trace(go.Scatter(
            x=list(range(1, len(h["test_loss"]) + 1)), y=h["test_loss"],
            name=f"#{r['run']}: {r['summary']}", mode="lines"))
    fig.update_layout(
        title=dict(text="Test loss of every run — lower and flatter is better",
                   font=dict(size=18)),
        xaxis_title="Epoch", yaxis_title="Test loss", height=420,
        hovermode="x unified", font=dict(size=14),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=10, r=10, t=60, b=10))
    return fig


def run_summary(r):
    parts = []
    if r["weight_decay"] > 0:
        parts.append(f"L2={r['weight_decay']:g}")
    if r["l1_lambda"] > 0:
        parts.append(f"L1={r['l1_lambda']:g}")
    if r["dropout_p"] > 0:
        parts.append(f"Dropout={r['dropout_p']:g}")
    if r["dropconnect_p"] > 0:
        parts.append(f"DropConnect={r['dropconnect_p']:g}")
    if r["batchnorm"]:
        parts.append("BatchNorm")
    return " + ".join(parts) if parts else "no regularization"


def prediction_figure(model, test_ds, class_names, device, n=12):
    model.eval()
    xs, ys = zip(*[test_ds[i] for i in range(n)])
    x = torch.stack(xs).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1).cpu().numpy()
    fig, axes = plt.subplots(2, n // 2, figsize=(2 * n // 2, 4.4))
    for ax, img, t, p in zip(axes.flat, xs, ys, pred):
        img = img.numpy()
        if img.shape[0] == 1:
            ax.imshow(img[0], cmap="gray")
        else:
            ax.imshow(np.transpose(img, (1, 2, 0)))
        ok = int(t) == int(p)
        ax.set_title(f"pred: {class_names[p]}\ntrue: {class_names[int(t)]}",
                     fontsize=8, color="green" if ok else "red")
        ax.axis("off")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Regularization Explorer", layout="wide")
st.title("Regularization Explorer")
st.caption(
    "Train a small neural network, switch regularization on or off, and watch "
    "what happens to **overfitting** — when a model scores well on data it "
    "has seen but poorly on data it hasn't."
)

with st.expander("New here? How to read this app"):
    st.markdown(
        """
1. Pick a dataset in the sidebar (or upload your own CSV) and press **Train**.
2. Look at the **gap** = train accuracy − test accuracy. A big gap means the
   model memorised the training data (overfitting).
3. Turn on a regularization technique, train again, and compare. The run
   table and the *All runs* chart keep every attempt so you can see which
   setting shrinks the gap without hurting test accuracy.

| Technique | What it does | What to expect |
|---|---|---|
| L2 weight decay | Penalises large weights | Smaller weights, smaller gap |
| L1 lambda | Pushes many weights to exactly 0 | Spike at 0 in the weight chart |
| Dropout | Randomly switches off neurons while training | Lower train accuracy, smaller gap |
| DropConnect | Randomly switches off individual weights | Like Dropout, finer-grained |
| BatchNorm | Normalises each layer's outputs | Faster training; not always less overfitting |
| Early stopping | Stops when test loss stops improving | Avoids the late-epoch rise in test loss |
"""
    )

for key, default in (("results", []), ("histories", [])):
    if key not in st.session_state:
        st.session_state[key] = default

csv_data = None
csv_ready = False

with st.sidebar:
    st.header("1. Data")
    dataset_name = st.selectbox(
        "Dataset", list(IMAGE_DATASETS) + [CSV_NAME],
        help="Image datasets download automatically the first time.")

    if dataset_name == CSV_NAME:
        up = st.file_uploader("CSV file (one row per example)", type=["csv"])
        test_frac = st.slider("Share of rows held out for testing", 0.1, 0.5, 0.3, 0.05)
        if up is not None:
            try:
                raw = pd.read_csv(up)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read that file as CSV: {e}")
                raw = None
            if raw is not None:
                target_col = st.selectbox(
                    "Target column (what to predict)", list(raw.columns),
                    index=len(raw.columns) - 1)
                try:
                    csv_data = prepare_csv(raw, target_col, test_frac, max_rows=50000)
                    csv_ready = True
                    st.success(f"{len(csv_data['train']):,} train / "
                               f"{len(csv_data['test']):,} test rows, "
                               f"{csv_data['in_dim']} features, "
                               f"{csv_data['n_classes']} classes")
                    for n in csv_data["notes"]:
                        st.caption(n)
                except ValueError as e:
                    st.error(str(e))
        else:
            st.info("Upload a CSV to begin. Tip: small datasets (a few hundred "
                    "rows) overfit easily — ideal for this demo.")
        train_size = None
    else:
        train_size = st.select_slider(
            "Training set size", options=[500, 2000, 10000], value=2000,
            help="Fewer training examples = more overfitting.")

    st.header("2. Regularization")
    weight_decay = st.number_input(
        "L2 weight decay", min_value=0.0, max_value=1.0, value=0.0,
        step=0.001, format="%.4f")
    l1_lambda = st.number_input(
        "L1 lambda", min_value=0.0, max_value=0.01, value=0.0,
        step=0.0001, format="%.5f")
    dropout_p = st.slider("Dropout rate", 0.0, 0.9, 0.0, 0.05)
    dropconnect_p = st.slider("DropConnect rate", 0.0, 0.9, 0.0, 0.05)
    batchnorm = st.checkbox("BatchNorm")
    early_stop = st.checkbox("Early stopping")

    st.header("3. Training")
    lr = st.select_slider("Learning rate", options=[0.0001, 0.001, 0.01, 0.1],
                          value=0.001)
    epochs = st.slider("Epochs", 5, 40, 20)
    run_button = st.button("Train", type="primary", width="stretch",
                           disabled=(dataset_name == CSV_NAME and not csv_ready))

    if dropout_p > 0 and dropconnect_p > 0:
        st.warning("Dropout and DropConnect are both on — turn one off "
                   "to keep the comparison clean, as in the lab.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if run_button:
    if dataset_name == CSV_NAME:
        data = csv_data
        train_ds, test_ds = data["train"], data["test"]
    else:
        data = load_image_dataset(dataset_name)
        train_ds = make_train_subset(data["train"], train_size)
        test_ds = data["test"]

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    set_seed()
    model = Net(in_dim=data["in_dim"], n_classes=data["n_classes"],
                dropout_p=dropout_p, batchnorm=batchnorm,
                dropconnect_p=dropconnect_p).to(device)

    progress_bar = st.progress(0, text="Training...")

    def cb(epoch, total):
        progress_bar.progress((epoch + 1) / total, text=f"Epoch {epoch + 1}/{total}")

    history = train_model(model, train_loader, test_loader, epochs, lr,
                          weight_decay, l1_lambda, device, early_stop, cb)
    progress_bar.empty()

    final_train_acc = history["train_acc"][-1]
    final_test_acc = history["test_acc"][-1]
    gap = final_train_acc - final_test_acc

    all_weights = torch.cat(
        [p.detach().flatten() for p in model.parameters() if p.dim() > 1])
    near_zero_frac = (all_weights.abs() < 1e-3).float().mean().item()

    row = {
        "run": len(st.session_state.results) + 1,
        "dataset": dataset_name,
        "train_size": len(train_ds),
        "weight_decay": weight_decay,
        "l1_lambda": l1_lambda,
        "dropout_p": dropout_p,
        "dropconnect_p": dropconnect_p,
        "batchnorm": batchnorm,
        "lr": lr,
        "epochs_ran": len(history["train_acc"]),
        "train_acc": round(final_train_acc, 4),
        "test_acc": round(final_test_acc, 4),
        "gap": round(gap, 4),
        "near_zero_frac": round(near_zero_frac, 4),
    }
    row["summary"] = run_summary(row)
    st.session_state.results.append(row)
    st.session_state.histories.append(history)
    st.session_state.last_weights = all_weights.cpu().numpy()
    st.session_state.last_preview = None
    if data["is_image"]:
        st.session_state.last_preview = prediction_figure(
            model, test_ds, data["class_names"], device)

# ----------------------------------------------------------------------
# Results for the current run
# ----------------------------------------------------------------------
if st.session_state.results:
    history = st.session_state.histories[-1]
    last = st.session_state.results[-1]

    st.subheader(f"Run #{last['run']}: {last['summary']}  ·  {last['dataset']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{last['train_acc']:.1%}",
              help="Accuracy on data the model trained on.")
    c2.metric("Test accuracy", f"{last['test_acc']:.1%}",
              help="Accuracy on data the model never saw — what really matters.")
    c3.metric("Generalization gap", f"{last['gap']:.1%}",
              help="Train − test accuracy. Big gap = overfitting.")

    gap = last["gap"]
    nums = (f"Train accuracy {last['train_acc']:.1%}, test accuracy "
            f"{last['test_acc']:.1%}")
    if gap > 0.10:
        st.warning(f"**Overfitting.** {nums}. The model does much better on "
                   "data it has seen than on new data. Try more dropout, L2, "
                   "or more training data.")
    elif last["train_acc"] < 0.70:
        st.info(f"**Underfitting.** {nums}. The model is not even learning the "
                "training data well. Try less regularization, more epochs, or "
                "a higher learning rate.")
    else:
        st.success(f"**Generalizing reasonably well.** {nums}. The gap is small.")

    tab_loss, tab_acc, tab_w, tab_pred = st.tabs(
        ["Loss", "Accuracy", "Weights", "Sample predictions"])
    with tab_loss:
        st.plotly_chart(curve_chart(history, "loss"), width="stretch")
        st.caption(
            "**How to read it:** the blue line is how wrong the model is on its "
            "training data, orange on unseen data. The shaded band is the gap. "
            "If the orange line turns upward while blue keeps falling, the "
            "model has started memorising — the green star marks the best epoch "
            "to stop at (early stopping).")
    with tab_acc:
        st.plotly_chart(curve_chart(history, "acc"), width="stretch")
        st.caption(
            "**How to read it:** same idea in plain percentages. Regularization "
            "usually lowers the blue line a little and narrows the shaded gap.")
    with tab_w:
        st.plotly_chart(weight_chart(st.session_state.last_weights,
                                     last["l1_lambda"], last["weight_decay"]),
                        width="stretch")
        st.caption(
            "**How to read it:** each bar counts how many of the network's "
            "weights have that value (log scale, so small bars are visible). "
            "L1 pushes many weights to exactly 0 → a tall spike in the middle. "
            "L2 shrinks all weights → a narrower bell. Unregularized models "
            "have the widest spread.")
    with tab_pred:
        if st.session_state.get("last_preview") is not None:
            st.pyplot(st.session_state.last_preview)
            st.caption("First 12 test images. Green title = correct, red = wrong.")
        else:
            st.info("Sample predictions are shown for image datasets only.")

# ----------------------------------------------------------------------
# All runs this session
# ----------------------------------------------------------------------
if st.session_state.results:
    st.subheader("Compare all runs")
    results = st.session_state.results
    if len(results) > 1:
        a, b = st.columns(2)
        a.plotly_chart(compare_chart(results), width="stretch")
        b.plotly_chart(overlay_chart(st.session_state.histories, results),
                       width="stretch")
    else:
        st.caption("Train at least two runs with different settings to compare them.")

    df = pd.DataFrame(results).drop(columns=["summary"])
    best_idx = df["test_acc"].idxmax()

    def highlight_best(row):
        return ["background-color: #d4f7d4; color: #111" if row.name == best_idx else ""
                for _ in row]

    st.dataframe(df.style.apply(highlight_best, axis=1), width="stretch")
    st.caption("Highlighted row = best test accuracy so far. "
               "near_zero_frac = share of weights with |w| < 0.001.")
    if st.button("Clear all runs"):
        st.session_state.results = []
        st.session_state.histories = []
        st.session_state.pop("last_weights", None)
        st.session_state.pop("last_preview", None)
        st.rerun()
else:
    st.info("Choose a dataset and press **Train** in the sidebar to start.")
