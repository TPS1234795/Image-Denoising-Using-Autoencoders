# ============================================================
# UNDERCOMPLETE AUTOENCODER - STREAMLIT DEPLOYMENT
# ============================================================

import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image
import os
import io


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Undercomplete Autoencoder",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🧠 Undercomplete Autoencoder")

st.subheader(
    "MNIST Image Reconstruction"
)

st.write(
    """
    Upload a handwritten digit image and the trained
    Undercomplete Autoencoder will reconstruct the image.
    """
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

WEIGHTS_PATH = "undercomplete_autoencoder.weights.h5"

INPUT_SIZE = 784
LATENT_DIM = 32


# ============================================================
# CREATE UNDERCOMPLETE AUTOENCODER
# ============================================================

def create_undercomplete_model():

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = keras.Input(
        shape=(784,),
        name="input"
    )

    # --------------------------------------------------------
    # Encoder
    # --------------------------------------------------------

    x = layers.Dense(
        256,
        activation="relu"
    )(inputs)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    # --------------------------------------------------------
    # Undercomplete latent representation
    # --------------------------------------------------------

    latent = layers.Dense(
        32,
        activation="relu",
        name="latent"
    )(x)

    # --------------------------------------------------------
    # Decoder
    # --------------------------------------------------------

    x = layers.Dense(
        128,
        activation="relu"
    )(latent)

    x = layers.Dense(
        256,
        activation="relu"
    )(x)

    outputs = layers.Dense(
        784,
        activation="sigmoid",
        name="output"
    )(x)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = keras.Model(
        inputs,
        outputs,
        name="Undercomplete_Autoencoder"
    )

    return model


# ============================================================
# LOAD WEIGHTS
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(
        WEIGHTS_PATH
    ):

        st.error(
            f"Model weights not found: {WEIGHTS_PATH}"
        )

        st.stop()

    # Create architecture
    model = create_undercomplete_model()

    # Load trained weights
    model.load_weights(
        WEIGHTS_PATH
    )

    return model


model = load_model()


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    st.write(
        "**Model:** Undercomplete Autoencoder"
    )

    st.write(
        "**Input:** 28 × 28 grayscale image"
    )

    st.write(
        "**Flattened input:** 784"
    )

    st.write(
        "**Latent dimension:** 32"
    )

    st.write(
        "**Parameters:** "
        f"{model.count_params():,}"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=[
        "png",
        "jpg",
        "jpeg"
    ]
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize to MNIST size
    image = image.resize(
        (28, 28)
    )

    # Convert to NumPy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Normalize
    image_array = (
        image_array / 255.0
    )

    return image_array


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # Open image
    original_image = Image.open(
        uploaded_file
    )

    # Display uploaded image
    st.subheader(
        "Uploaded Image"
    )

    st.image(
        original_image,
        width=300
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed_image = preprocess_image(
        original_image
    )

    # --------------------------------------------------------
    # Flatten 28×28 → 784
    # --------------------------------------------------------

    input_data = processed_image.reshape(
        1,
        784
    )

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    reconstructed = model.predict(
        input_data,
        verbose=0
    )

    # --------------------------------------------------------
    # Reshape 784 → 28×28
    # --------------------------------------------------------

    reconstructed_image = (
        reconstructed[0]
        .reshape(28, 28)
    )

    reconstructed_image = np.clip(
        reconstructed_image,
        0,
        1
    )


    # ========================================================
    # CALCULATE ERROR
    # ========================================================

    mse = np.mean(
        (
            processed_image -
            reconstructed_image
        ) ** 2
    )

    mae = np.mean(
        np.abs(
            processed_image -
            reconstructed_image
        )
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original Image"
        )

        st.image(
            processed_image,
            width=300
        )

    with col2:

        st.subheader(
            "Reconstructed Image"
        )

        st.image(
            reconstructed_image,
            width=300
        )


    # ========================================================
    # METRICS
    # ========================================================

    st.divider()

    st.subheader(
        "Reconstruction Metrics"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "MSE",
            f"{mse:.6f}"
        )

    with col2:

        st.metric(
            "MAE",
            f"{mae:.6f}"
        )


    # ========================================================
    # DOWNLOAD RECONSTRUCTED IMAGE
    # ========================================================

    reconstructed_uint8 = (
        reconstructed_image * 255
    ).astype(
        np.uint8
    )

    reconstructed_pil = Image.fromarray(
        reconstructed_uint8
    )

    buffer = io.BytesIO()

    reconstructed_pil.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        label="⬇️ Download Reconstructed Image",
        data=buffer.getvalue(),
        file_name="reconstructed_image.png",
        mime="image/png"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MNIST | Undercomplete Autoencoder | "
    "TensorFlow + Keras + Streamlit"
)