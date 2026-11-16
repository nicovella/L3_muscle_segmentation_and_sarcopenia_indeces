"""L3 muscle segmentation — simple local interface.

Same logic as U-Net_application.ipynb, as a web page instead of a notebook:
drop a .nii / .nii.gz L3 CT slice, see the CT with the muscle overlay,
get area in cm2 and mean HU for total muscle and psoas, download the
predicted mask (.nii) and the text report (.txt).

Runs only on your own computer. Nothing is uploaded anywhere.

To run (in the VS Code terminal, in this folder):
    streamlit run app.py

Files needed in this same folder:
    app.py
    unet_sarcopenia_best.keras   <- retrained weights (Second_Training)
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tempfile
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import streamlit as st
import tensorflow as tf
from matplotlib.patches import Patch

MODEL_PATH = Path(__file__).parent / "unet_sarcopenia_best.keras"
HU_MIN, HU_MAX = -30.0, 150.0
TARGET_SIZE = (256, 256)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def segment(ct_data, header, model):
    """Run the U-Net on one 2D slice. Returns the mask at native resolution."""
    img = np.squeeze(ct_data).astype(np.float32)
    img = np.clip(img, HU_MIN, HU_MAX)
    img = (img - HU_MIN) / (HU_MAX - HU_MIN)

    img_tensor = tf.expand_dims(tf.convert_to_tensor(img, dtype=tf.float32), axis=-1)
    img_resized = tf.image.resize(img_tensor, TARGET_SIZE, method="bilinear").numpy()
    preds = model.predict(np.expand_dims(img_resized, axis=0), verbose=0)
    pred_mask_256 = np.argmax(preds, axis=-1)[0]

    # Nearest-neighbour upsampling: keeps mask labels exactly 0, 1, 2.
    mask = tf.image.resize(
        np.expand_dims(pred_mask_256, axis=-1),
        img.shape,
        method="nearest",
    ).numpy().squeeze()
    return mask


def measure(mask, ct_data, header):
    """Area (cm2) and mean HU for total muscle (1+2) and psoas (2). No L4."""
    mask_total = np.logical_or(mask == 1, mask == 2)
    mask_psoas = mask == 2
    px = float(header["pixdim"][1])
    py = float(header["pixdim"][2])
    pixel_area_cm2 = px * py / 100.0

    total_area = float(np.count_nonzero(mask_total) * pixel_area_cm2)
    psoas_area = float(np.count_nonzero(mask_psoas) * pixel_area_cm2)
    total_hu = float(np.mean(ct_data[mask_total])) if np.any(mask_total) else 0.0
    psoas_hu = float(np.mean(ct_data[mask_psoas])) if np.any(mask_psoas) else 0.0
    return total_area, total_hu, psoas_area, psoas_hu


def overlay_figure(ct_slice, mask):
    cmap_masks = mcolors.ListedColormap(["none", "#1f77b4", "#2ca02c"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(ct_slice, cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")
    axes[1].imshow(ct_slice, cmap="gray")
    masked = np.ma.masked_where(mask == 0, mask)
    axes[1].imshow(masked, cmap=cmap_masks, alpha=0.6, vmin=0, vmax=2)
    axes[1].set_title("Prediction: blue = abdominals, green = psoas")
    axes[1].axis("off")
    fig.legend(
        handles=[
            Patch(facecolor="#1f77b4", alpha=0.6, label="Abdominals (1)"),
            Patch(facecolor="#2ca02c", alpha=0.6, label="Psoas (2)"),
        ],
        loc="lower center",
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return fig


st.set_page_config(page_title="L3 Muscle Segmentation", layout="wide")
st.title("L3 Muscle Segmentation")
st.write(
    "Drop an L3 CT slice (`.nii` or `.nii.gz`) below. "
    "The retrained U-Net segments abdominals and psoas and gives you "
    "area in cm² and mean density in HU."
)

if not MODEL_PATH.exists():
    st.error(
        f"Weights file not found: {MODEL_PATH.name}. "
        "Copy unet_sarcopenia_best.keras into this same folder and reload the page."
    )
    st.stop()

model = load_model()

height_m = st.number_input(
    "Height in metres (optional — only needed to calculate SMI)",
    min_value=0.0,
    max_value=2.5,
    value=0.0,
    step=0.01,
    format="%.2f",
)

uploaded = st.file_uploader(
    "Drag and drop your CT slice here, or click to select a file",
    type=["nii", "gz"],
)

if uploaded is None:
    st.info("No file yet — drop a .nii or .nii.gz file above to start.")
    st.stop()

# nibabel needs a real file, so save the upload to a temporary file first.
suffix = ".nii.gz" if uploaded.name.endswith(".gz") else ".nii"
with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
    tmp.write(uploaded.getvalue())
    tmp_path = Path(tmp.name)

try:
    nii_img = nib.load(tmp_path)
    ct_data = nii_img.get_fdata()
    ct_slice = np.squeeze(ct_data)
    if ct_slice.ndim != 2:
        st.error(
            f"This file has shape {ct_data.shape} — the app expects one 2D L3 slice, "
            "for example 512x512x1."
        )
        st.stop()

    with st.spinner("Segmenting…"):
        mask = segment(ct_data, nii_img.header, model)
    total_area, total_hu, psoas_area, psoas_hu = measure(mask, ct_slice, nii_img.header)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("Total muscle (1+2) — area", f"{total_area:.1f} cm²")
    col1.metric("Total muscle (1+2) — mean density", f"{total_hu:.0f} HU")
    col2.metric("Psoas (2) — area", f"{psoas_area:.1f} cm²")
    col2.metric("Psoas (2) — mean density", f"{psoas_hu:.0f} HU")

    if height_m > 0:
        smi = total_area / (height_m ** 2)
        st.metric("SMI (total muscle area / height²)", f"{smi:.1f} cm²/m²")

    st.pyplot(overlay_figure(ct_slice, mask))

    # Build the mask file and the text report for download.
    patient_id = uploaded.name.replace(".nii", "").replace(".gz", "")
    mask_nii = nib.Nifti1Image(
        mask.astype(np.int16), affine=nii_img.affine, header=nii_img.header
    )
    with tempfile.NamedTemporaryFile(suffix=".nii", delete=False) as tmp_mask:
        nib.save(mask_nii, tmp_mask.name)
        mask_bytes = Path(tmp_mask.name).read_bytes()
    Path(tmp_mask.name).unlink(missing_ok=True)

    report_lines = [
        "CLINICAL SARCOPENIA ASSESSMENT REPORT (AI-Generated)",
        "",
        f"Patient ID: {patient_id}",
        f"Source File: {uploaded.name}",
        "--------------------------------------------------",
        "L3 Total Muscle (Class 1+2):",
        f"  -> Area: {total_area:.1f} cm²",
        f"  -> Mean Density: {total_hu:.0f} HU",
        "",
        "L3 Psoas Muscle (Class 2):",
        f"  -> Area: {psoas_area:.1f} cm²",
        f"  -> Mean Density: {psoas_hu:.0f} HU",
    ]
    if height_m > 0:
        report_lines += [
            "",
            f"SMI (total muscle area / height²): {total_area / (height_m ** 2):.1f} cm²/m²",
            f"Height used: {height_m:.2f} m",
        ]
    report_lines += [
        "--------------------------------------------------",
        "Report generated automatically via Custom U-Net Pipeline.",
    ]
    report_text = "\n".join(report_lines)

    st.download_button(
        "Download predicted mask (.nii)",
        data=mask_bytes,
        file_name=f"predicted_mask_{uploaded.name.replace('.gz', '')}",
        mime="application/octet-stream",
    )
    st.download_button(
        "Download report (.txt)",
        data=report_text.encode("utf-8"),
        file_name=f"predicted_mask_{patient_id}_report.txt",
        mime="text/plain",
    )
finally:
    tmp_path.unlink(missing_ok=True)

