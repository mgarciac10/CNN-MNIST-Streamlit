"""
MNIST Digit Classifier — Streamlit App
- Trains a CNN on the MNIST dataset
- Provides a drawable canvas for real-time digit classification
"""

import io
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────
# 1.  CNN Architecture
# ─────────────────────────────────────────────
class MNISTNet(nn.Module):
    """
    Compact CNN:
      Conv(1→32, 3) → ReLU → Conv(32→64, 3) → ReLU → MaxPool → Dropout
      → Conv(64→64, 3) → ReLU → MaxPool → Dropout
      → FC(1600→128) → ReLU → Dropout → FC(128→10)
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─────────────────────────────────────────────
# 2.  Training helpers
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_data(batch_size: int = 64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return correct / total


# ─────────────────────────────────────────────
# 3.  Inference helper
# ─────────────────────────────────────────────
def predict_digit(model, canvas_image: np.ndarray):
    """
    Takes the RGBA canvas image → grayscale 28×28 tensor → returns
    (predicted_digit, probability_array).
    """
    img = Image.fromarray(canvas_image.astype("uint8"), "RGBA").convert("L")

    # The canvas draws white on black — MNIST is white-on-black too,
    # so we just resize and normalise.
    img = img.resize((28, 28), Image.LANCZOS)

    # Centre the digit using centre-of-mass shift (mimics MNIST preprocessing)
    arr = np.array(img, dtype=np.float32)
    if arr.sum() == 0:
        return None, None

    # Normalise to 0-1 then apply MNIST stats
    arr = arr / 255.0
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor = (tensor - 0.1307) / 0.3081
    tensor = tensor.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return int(probs.argmax()), probs


# ─────────────────────────────────────────────
# 4.  Streamlit UI
# ─────────────────────────────────────────────
def main():
    st.set_page_config(page_title="MNIST CNN Classifier", layout="wide", page_icon="🔢")

    # ── Custom CSS ──────────────────────────
    st.markdown("""
    <style>
        .block-container { max-width: 1100px; padding-top: 2rem; }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            color: white;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .metric-card h2 { margin: 0; font-size: 2rem; }
        .metric-card p  { margin: 0; font-size: 0.85rem; opacity: 0.85; }
        .prediction-box {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            color: white;
            margin: 1rem 0;
        }
        .prediction-box h1 { font-size: 5rem; margin: 0; }
        .prediction-box p  { font-size: 1rem; opacity: 0.9; }
        .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
        .stTabs [data-baseweb="tab"] {
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──────────────────────────────
    st.markdown("# 🔢 MNIST Digit Classifier")
    st.markdown("Train a CNN on MNIST, then draw a digit and watch the model classify it in real time.")

    tab_train, tab_draw = st.tabs(["🏋️ Train Model", "✏️ Draw & Classify"])

    # ══════════════════════════════════════════
    #   TAB 1 — TRAINING
    # ══════════════════════════════════════════
    with tab_train:
        col_cfg, col_status = st.columns([1, 2])

        with col_cfg:
            st.markdown("### Configuration")
            epochs = st.slider("Epochs", 1, 20, 5)
            lr = st.select_slider("Learning rate", options=[0.01, 0.005, 0.001, 0.0005, 0.0001], value=0.001)
            batch_size = st.selectbox("Batch size", [32, 64, 128, 256], index=1)
            train_btn = st.button("🚀 Start training", use_container_width=True, type="primary")

        with col_status:
            if train_btn:
                train_loader, test_loader = load_data(batch_size)
                model = MNISTNet().to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()

                history = {"loss": [], "acc": [], "val_acc": []}
                progress = st.progress(0, text="Preparing…")
                metric_cols = st.columns(3)
                chart_placeholder = st.empty()

                for epoch in range(1, epochs + 1):
                    progress.progress(epoch / epochs, text=f"Epoch {epoch}/{epochs}")
                    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
                    val_acc = evaluate(model, test_loader)
                    history["loss"].append(loss)
                    history["acc"].append(acc)
                    history["val_acc"].append(val_acc)

                    with metric_cols[0]:
                        st.markdown(
                            f'<div class="metric-card"><h2>{loss:.4f}</h2><p>Train Loss</p></div>',
                            unsafe_allow_html=True,
                        )
                    with metric_cols[1]:
                        st.markdown(
                            f'<div class="metric-card"><h2>{acc:.2%}</h2><p>Train Acc</p></div>',
                            unsafe_allow_html=True,
                        )
                    with metric_cols[2]:
                        st.markdown(
                            f'<div class="metric-card"><h2>{val_acc:.2%}</h2><p>Test Acc</p></div>',
                            unsafe_allow_html=True,
                        )

                    # Live training chart
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
                    ep_range = range(1, epoch + 1)
                    ax1.plot(ep_range, history["loss"], "o-", color="#667eea", linewidth=2)
                    ax1.set_title("Loss", fontweight="bold")
                    ax1.set_xlabel("Epoch")
                    ax1.grid(True, alpha=0.3)

                    ax2.plot(ep_range, [a * 100 for a in history["acc"]], "o-", color="#11998e", linewidth=2, label="Train")
                    ax2.plot(ep_range, [a * 100 for a in history["val_acc"]], "s--", color="#764ba2", linewidth=2, label="Test")
                    ax2.set_title("Accuracy (%)", fontweight="bold")
                    ax2.set_xlabel("Epoch")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    fig.tight_layout()
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)

                progress.progress(1.0, text="✅ Training complete!")
                st.session_state["model"] = model
                st.success(f"Model ready — Test accuracy: **{val_acc:.2%}**. Head to the *Draw & Classify* tab!")

            elif "model" not in st.session_state:
                st.info("Configure parameters on the left and click **Start training** to begin.")
            else:
                st.success("✅ Model is trained and ready. Switch to the **Draw & Classify** tab!")

    # ══════════════════════════════════════════
    #   TAB 2 — DRAW & CLASSIFY
    # ══════════════════════════════════════════
    with tab_draw:
        if "model" not in st.session_state:
            st.warning("⚠️ Train the model first in the **Train Model** tab.")
            return

        model: MNISTNet = st.session_state["model"]

        col_canvas, col_result = st.columns([1, 1])

        with col_canvas:
            st.markdown("### Draw a digit (0-9)")
            st.caption("Use your mouse or finger to draw on the black canvas below.")

            stroke_width = st.slider("Brush size", 10, 40, 22)

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas",
                display_toolbar=True,
            )

        with col_result:
            st.markdown("### Prediction")

            if canvas_result.image_data is not None:
                digit, probs = predict_digit(model, canvas_result.image_data)

                if digit is not None:
                    confidence = probs[digit] * 100
                    st.markdown(
                        f'<div class="prediction-box">'
                        f'<h1>{digit}</h1>'
                        f'<p>Confidence: {confidence:.1f}%</p>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    # Bar chart of all probabilities
                    st.markdown("#### Class probabilities")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    colors = ["#38ef7d" if i == digit else "#667eea" for i in range(10)]
                    bars = ax.bar(range(10), probs * 100, color=colors, edgecolor="white", linewidth=0.5)
                    ax.set_xticks(range(10))
                    ax.set_xlabel("Digit")
                    ax.set_ylabel("Probability (%)")
                    ax.set_ylim(0, 105)
                    ax.grid(axis="y", alpha=0.3)
                    for bar, p in zip(bars, probs * 100):
                        if p > 3:
                            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                                    f"{p:.1f}", ha="center", va="bottom", fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                    # Show what the model "sees"
                    with st.expander("🔍 What the model sees (28×28 input)"):
                        preview = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA").convert("L")
                        preview = preview.resize((28, 28), Image.LANCZOS)
                        fig2, ax2 = plt.subplots(figsize=(2, 2))
                        ax2.imshow(np.array(preview), cmap="gray")
                        ax2.axis("off")
                        fig2.tight_layout()
                        st.pyplot(fig2)
                        plt.close(fig2)
                else:
                    st.markdown("*Draw something on the canvas to see the prediction.*")
            else:
                st.markdown("*Draw something on the canvas to see the prediction.*")


if __name__ == "__main__":
    main()
