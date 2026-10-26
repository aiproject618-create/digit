
import io
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered"
)

st.title("✍️ Handwritten Digit Recognition")
st.write("CBSE Class 12 AI Project (No training required)")

# ---------------- LOAD BUILT-IN MODEL ----------------
@st.cache_resource
def load_knn():
    digits = load_digits()
    X = digits.data
    y = digits.target

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)   # Uses built-in dataset automatically
    return model

model = load_knn()

# ---------------- PREPROCESS IMAGE ----------------
def preprocess(image):
    image = ImageOps.exif_transpose(image).convert("L")

    image = ImageOps.invert(image)
    image = image.resize((8, 8))

    arr = np.asarray(image, dtype=np.float32)

    arr = (arr / 255.0) * 16.0
    arr = arr.reshape(1, -1)

    return arr, image

# ---------------- UPLOAD ----------------
uploaded = st.file_uploader(
    "Upload a handwritten digit (0–9)",
    type=["png", "jpg", "jpeg"]
)

if uploaded:
    image = Image.open(io.BytesIO(uploaded.read()))

    st.image(image, caption="Uploaded Image", width=220)

    X, processed = preprocess(image)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    st.success(f"Predicted Digit: **{prediction}**")

    st.metric(
        "Confidence",
        f"{max(probabilities)*100:.1f}%"
    )

    st.subheader("Processed Image (8×8)")
    st.image(processed, width=140)

    st.subheader("Digit Probabilities")

    for i, p in enumerate(probabilities):
        st.write(f"Digit {i}: {p*100:.1f}%")
        st.progress(float(p))

st.divider()
st.caption("Built using Streamlit + Scikit-learn")
