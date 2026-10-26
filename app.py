import os
import io
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import matplotlib.pyplot as plt

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DigitLab | Handwritten Digit AI",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "digit_model.keras"

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0b1020 0%, #111b35 55%, #101827 100%); }
    [data-testid="stHeader"] { background: transparent; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 22px;
        background: linear-gradient(120deg, rgba(64,96,255,.28), rgba(0,210,190,.12));
        border: 1px solid rgba(160,180,255,.22); margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 2.4rem; color: #f4f7ff; }
    .hero p { color: #c5d0eb; margin: .45rem 0 0 0; font-size: 1.03rem; }
    .small-label { color: #9eafd4; text-transform: uppercase; letter-spacing: .12em; font-size: .78rem; }
    .result-card {
        padding: 1.25rem; border-radius: 18px; background: rgba(255,255,255,.06);
        border: 1px solid rgba(255,255,255,.12); text-align: center;
    }
    .digit { font-size: 5rem; line-height: 1.1; font-weight: 800; color: #73f2d0; }
    .muted { color: #b7c4df; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.055); padding: 14px 16px;
        border-radius: 14px; border: 1px solid rgba(255,255,255,.10);
    }
    .stButton button, .stDownloadButton button { border-radius: 12px; font-weight: 650; }
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model(path, modified_time):
    # modified_time invalidates Streamlit's cache if the model file changes.
    return tf.keras.models.load_model(path, compile=False)

st.markdown("""
<div class="hero">
  <div class="small-label">Artificial Intelligence • Computer Vision</div>
  <h1>✍️ DigitLab</h1>
  <p>Draw a digit. Let a neural network recognize it. Explore the prediction and confidence across all ten classes.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.error("The trained model file `digit_model.keras` was not found.")
    st.markdown("""
**First-time setup**
1. Run `python train_model.py` locally, or follow the Google Colab instructions in `README.md`.
2. Place the generated `digit_model.keras` beside `app.py`.
3. Commit the model file to your GitHub repository and redeploy Streamlit Cloud.
""")
    st.stop()

try:
    model = load_model(MODEL_PATH, os.path.getmtime(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load `digit_model.keras`: {exc}")
    st.info("Train or regenerate the model using `train_model.py`.")
    st.stop()

# ---------------- PREPROCESSING ----------------
def prepare_digit(pil_image):
    """Convert a handwritten digit image to MNIST-like 28x28 grayscale input."""
    image = ImageOps.exif_transpose(pil_image).convert("L")
    arr = np.asarray(image, dtype=np.uint8)

    # Decide whether to invert: MNIST has a dark background and bright strokes.
    # If most pixels are bright, assume a white-paper image and invert it.
    if float(arr.mean()) > 127:
        arr = 255 - arr

    # Remove faint noise, then crop around visible strokes.
    arr = np.where(arr > 35, arr, 0).astype(np.uint8)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32), None

    cropped = Image.fromarray(arr).crop(
        (max(0, xs.min()-2), max(0, ys.min()-2),
         min(arr.shape[1], xs.max()+3), min(arr.shape[0], ys.max()+3))
    )

    # Fit inside 20x20, leaving a 4-pixel margin on each side.
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(cropped, ((28-new_size[0]) // 2, (28-new_size[1]) // 2))

    # Slight blur reduces harsh edges; normalize to 0..1 as in training.
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=0.25))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.reshape(1, 28, 28, 1), canvas

def predict_digit(batch):
    probs = model.predict(batch, verbose=0)[0]
    order = np.argsort(probs)[::-1]
    return int(order[0]), float(probs[order[0]]), probs, order

def show_results(batch, processed_image, source_name):
    digit, confidence, probs, order = predict_digit(batch)
    col1, col2, col3 = st.columns([1, 1, 1.5], gap="large")

    with col1:
        st.markdown(
            f'<div class="result-card"><div class="small-label">Predicted digit</div>'
            f'<div class="digit">{digit}</div><div class="muted">{confidence*100:.1f}% confidence</div></div>',
            unsafe_allow_html=True
        )
        st.write("")
        if confidence >= 0.80:
            st.success("Strong model confidence")
        elif confidence >= 0.50:
            st.warning("Moderate confidence — check the input")
        else:
            st.info("Low confidence — try drawing or uploading a clearer digit")

    with col2:
        st.markdown("**Model input (28 × 28)**")
        if processed_image is not None:
            st.image(processed_image, width=180, caption="Processed grayscale image")
        st.caption(f"Input source: {source_name}")

    with col3:
        st.markdown("**Probability by digit**")
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        ax.bar(np.arange(10), probs)
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    top3 = order[:3]
    st.markdown("**Top 3 predictions**")
    top_cols = st.columns(3)
    for col, idx in zip(top_cols, top3):
        col.metric(f"Digit {int(idx)}", f"{probs[idx]*100:.2f}%")

    with st.expander("View all class probabilities"):
        for i, p in enumerate(probs):
            st.write(f"**{i}** — {p*100:.2f}%")
            st.progress(float(p))

    st.caption(
        "Confidence is the model's probability output, not a guarantee. "
        "Performance can vary with handwriting style, image quality, and preprocessing."
    )

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("About this project")
    st.write("**Model:** Convolutional Neural Network (CNN)")
    st.write("**Dataset:** MNIST handwritten digits")
    st.write("**Classes:** 0–9")
    st.write("**Input size:** 28 × 28 grayscale")
    st.divider()
    st.markdown("**How it works**")
    st.write("1. Draw or upload a digit")
    st.write("2. Convert it to grayscale")
    st.write("3. Crop and resize to 28 × 28")
    st.write("4. Run CNN prediction")
    st.write("5. Compare probabilities")

# ---------------- INPUT TABS ----------------
draw_tab, upload_tab, demo_tab = st.tabs(["✍️ Draw a digit", "🖼️ Upload an image", "🧪 Try a demo"])

with draw_tab:
    st.subheader("Draw one digit")
    st.write("Use a dark background and a bright stroke. Draw one large digit in the middle.")
    if not CANVAS_AVAILABLE:
        st.warning("Drawing canvas is unavailable. Use the Upload tab instead.")
    else:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="digit_canvas",
            update_streamlit=True,
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            draw_clicked = st.button("Recognize drawn digit", type="primary", use_container_width=True)
        with c2:
            st.button("Clear canvas", use_container_width=True, on_click=lambda: st.session_state.update({"digit_canvas": None}))
        if draw_clicked:
            if canvas_result.image_data is None:
                st.warning("Draw a digit first.")
            else:
                rgba = np.asarray(canvas_result.image_data).astype(np.uint8)
                if rgba[:, :, 3].max() == 0:
                    st.warning("The canvas is blank. Draw a digit first.")
                else:
                    # Canvas background is black and strokes are white.
                    rgb = rgba[:, :, :3]
                    gray = Image.fromarray(rgb).convert("L")
                    batch, processed = prepare_digit(gray)
                    if processed is None:
                        st.warning("No visible digit found. Try drawing more clearly.")
                    else:
                        show_results(batch, processed, "Drawing canvas")

with upload_tab:
    st.subheader("Upload a handwritten digit image")
    uploaded = st.file_uploader(
        "Choose PNG, JPG or JPEG",
        type=["png", "jpg", "jpeg", "webp"],
        key="upload_digit"
    )
    if uploaded is not None:
        try:
            image = Image.open(io.BytesIO(uploaded.getvalue()))
            st.image(image, caption="Original upload", width=260)
            if st.button("Recognize uploaded digit", type="primary"):
                batch, processed = prepare_digit(image)
                if processed is None:
                    st.error("No clear foreground digit was found. Try a darker background or a clearer image.")
                else:
                    show_results(batch, processed, uploaded.name)
        except Exception:
            st.error("This file could not be read as an image. Please try a valid PNG or JPG.")

with demo_tab:
    st.subheader("Test with a sample digit")
    st.write("This uses a synthetic sample, so it is only a quick pipeline check.")
    demo_digit = st.select_slider("Choose sample digit", options=list(range(10)), value=5)
    if st.button("Recognize sample", type="primary"):
        # Use a PIL default font to draw a sample on a dark background.
        from PIL import ImageDraw, ImageFont
        sample = Image.new("L", (140, 140), 0)
        draw = ImageDraw.Draw(sample)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 110)
        except OSError:
            font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), str(demo_digit), font=font, stroke_width=1)
        tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        draw.text(((140-tw)//2-bbox[0], (140-th)//2-bbox[1]), str(demo_digit),
                  fill=255, font=font, stroke_width=1)
        st.image(sample, caption="Synthetic sample", width=160)
        batch, processed = prepare_digit(sample)
        show_results(batch, processed, f"Synthetic digit {demo_digit}")

st.divider()
st.caption("DigitLab • Educational demonstration for CBSE Class 12 AI")
