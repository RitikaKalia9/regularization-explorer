"""
Regularization Explorer — Lab 4 Home Assignment (Neural Networks Lab, CCET)

A Streamlit app that lets you configure a small FashionMNIST classifier
(the same 784 -> 256 -> 10 network from Lab 4), train it with any mix of
L2 weight decay, L1 penalty, Dropout, DropConnect and BatchNorm, and
compare runs side by side.

Run with:
    streamlit run app.py
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

SEED = 42


def set_seed(seed=SEED):
    """Reset the seed before every model so runs are comparable (Q: 'why
    did we reset the seed before building every model?')."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Downloading FashionMNIST...")
def load_datasets():
    transform = transforms.ToTensor()
    train_full = torchvision.datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_full = torchvision.datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )
    return train_full, test_full


def make_train_subset(train_full, train_size):
    set_seed()
    indices = torch.randperm(len(train_full))[:train_size]
    return Subset(train_full, indices)


# ----------------------------------------------------------------------
# Model — DropConnect layer + the Lab 4 network with optional
# Dropout / BatchNorm / DropConnect
# ----------------------------------------------------------------------
class DropConnectLinear(nn.Module):
    """Dropout's cousin: masks individual weights instead of whole
    neurons. Mask shape = weight matrix shape (256 x 784), not the
    neuron list (256), which is what makes this DropConnect rather
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
    def __init__(self, dropout_p=0.0, batchnorm=False, dropconnect_p=0.0,
                 hidden=256):
        super().__init__()
        if dropconnect_p > 0:
            self.fc1 = DropConnectLinear(784, hidden, p=dropconnect_p)
        else:
            self.fc1 = nn.Linear(784, hidden)
        self.bn = nn.BatchNorm1d(hidden) if batchnorm else None
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else None
        self.fc2 = nn.Linear(hidden, 10)

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
# UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Regularization Explorer", layout="wide")
st.title("Regularization Explorer")
st.caption(
    "Lab 4 mini project — set up a network, train it, and compare it "
    "with earlier runs."
)

if "results" not in st.session_state:
    st.session_state.results = []

with st.sidebar:
    st.header("Settings")
    train_size = st.select_slider(
        "Training set size (extra credit)", options=[500, 2000, 10000], value=2000
    )
    weight_decay = st.number_input(
        "L2 weight decay", min_value=0.0, max_value=1.0, value=0.0,
        step=0.001, format="%.4f"
    )
    l1_lambda = st.number_input(
        "L1 lambda", min_value=0.0, max_value=0.01, value=0.0,
        step=0.0001, format="%.5f"
    )
    dropout_p = st.slider("Dropout rate", 0.0, 0.9, 0.0, 0.05)
    dropconnect_p = st.slider(
        "DropConnect rate (extra credit)", 0.0, 0.9, 0.0, 0.05
    )
    batchnorm = st.checkbox("BatchNorm")
    lr = st.select_slider(
        "Learning rate", options=[0.0001, 0.001, 0.01, 0.1], value=0.001
    )
    epochs = st.slider("Epochs", 5, 40, 20)
    early_stop = st.checkbox("Early stopping (extra credit)")
    run_button = st.button("Train", type="primary")

    if dropout_p > 0 and dropconnect_p > 0:
        st.warning("Dropout and DropConnect are both on — turn one off "
                    "to keep the comparison clean, as in the lab.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if run_button:
    train_full, test_full = load_datasets()
    train_subset = make_train_subset(train_full, train_size)
    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_full, batch_size=256, shuffle=False)

    set_seed()
    model = Net(dropout_p=dropout_p, batchnorm=batchnorm,
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
        [p.detach().flatten() for p in model.parameters() if p.dim() > 1]
    )
    near_zero_frac = (all_weights.abs() < 1e-3).float().mean().item()

    st.session_state.results.append({
        "run": len(st.session_state.results) + 1,
        "train_size": train_size,
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
    })
    st.session_state.last_history = history
    st.session_state.last_weights = all_weights.cpu().numpy()

# ----------------------------------------------------------------------
# Results for the current run
# ----------------------------------------------------------------------
if "last_history" in st.session_state:
    history = st.session_state.last_history
    last = st.session_state.results[-1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Train accuracy", f"{last['train_acc']:.3f}")
    c2.metric("Test accuracy", f"{last['test_acc']:.3f}")
    c3.metric("Generalization gap", f"{last['gap']:.3f}")

    gap = last["gap"]
    if gap > 0.10:
        verdict = (f"training accuracy {last['train_acc']:.2f}, test accuracy "
                   f"{last['test_acc']:.2f} — this model is overfitting, "
                   f"try more dropout or L2.")
    elif last["train_acc"] < 0.70:
        verdict = (f"training accuracy {last['train_acc']:.2f}, test accuracy "
                   f"{last['test_acc']:.2f} — this model is underfitting, "
                   f"try less regularization or more capacity.")
    else:
        verdict = (f"training accuracy {last['train_acc']:.2f}, test accuracy "
                   f"{last['test_acc']:.2f} — this model is generalizing "
                   f"reasonably well.")
    st.info(verdict)

    fig, ax = plt.subplots()
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["test_loss"], label="Test loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs test loss")
    ax.legend()
    st.pyplot(fig)

    if "last_weights" in st.session_state:
        fig2, ax2 = plt.subplots()
        ax2.hist(st.session_state.last_weights, bins=100)
        ax2.set_title("Weight histogram (extra credit) — L1 zeros show up as a spike at 0")
        ax2.set_xlabel("Weight value")
        st.pyplot(fig2)

# ----------------------------------------------------------------------
# All runs this session
# ----------------------------------------------------------------------
if st.session_state.results:
    st.subheader("All runs this session")
    df = pd.DataFrame(st.session_state.results)
    best_idx = df["test_acc"].idxmax()

    def highlight_best(row):
        return ["background-color: #d4f7d4" if row.name == best_idx else ""
                for _ in row]

    st.dataframe(df.style.apply(highlight_best, axis=1), use_container_width=True)
    st.caption("Highlighted row = best test accuracy so far.")