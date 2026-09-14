"""
Industrial PCB Defect Detection — Streamlit Demo
Role: Application & Integration (Saif)

Run with:
    streamlit run app/app.py
"""

import io
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# Allow importing from src/ regardless of where streamlit is launched from
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import load_model, run_inference, load_model_stats, CLASS_COLORS  # noqa: E402
from src.utils import is_supported_image  # noqa: E402

# ----------------------------------------------------------------------------
# Page config & global style
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    #MainMenu, footer {visibility: hidden;}
    .app-header {
        padding: 1.1rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: #f8fafc;
        margin-bottom: 1.2rem;
    }
    .app-header h1 { margin: 0; font-size: 1.6rem; }
    .app-header p { margin: .25rem 0 0 0; color: #cbd5e1; font-size: .92rem; }

    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: .8rem 1rem;
        text-align: center;
    }
    .metric-card .val { font-size: 1.4rem; font-weight: 700; color: #0f172a; }
    .metric-card .lbl { font-size: .78rem; color: #64748b; text-transform: uppercase; letter-spacing: .04em; }

    .class-chip {
        display: inline-flex; align-items: center; gap: .4rem;
        padding: .2rem .6rem; margin: .15rem .2rem .15rem 0;
        border-radius: 999px; font-size: .8rem; background: #f1f5f9;
    }
    .class-chip .dot { width: .6rem; height: .6rem; border-radius: 50%; display: inline-block; }

    .footer-note { text-align: center; color: #94a3b8; font-size: .78rem; margin-top: 2rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>🔍 Industrial PCB Defect Detection</h1>
        <p>Upload one or more PCB images to detect and localize manufacturing defects using a fine-tuned YOLO11n model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model():
    return load_model()


# ----------------------------------------------------------------------------
# Sidebar — model card, controls, legend
# ----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.05, 1.0, 0.25, 0.05)
    iou_threshold = st.slider("IoU threshold (NMS)", 0.05, 1.0, 0.45, 0.05)

    st.markdown("---")
    st.subheader("🧠 Model Card")
    stats = load_model_stats()
    st.caption("YOLO11n · 320×320 · held-out test evaluation")
    if stats:
        c1, c2 = st.columns(2)
        c1.metric("mAP50", stats.get("mAP50", "—"))
        c2.metric("mAP50-95", stats.get("mAP50-95", "—"))
        c3, c4 = st.columns(2)
        c3.metric("Precision", stats.get("Precision", "—"))
        c4.metric("Recall", stats.get("Recall", "—"))
        st.caption(f"Metrics split: {stats.get('Split', 'Test')} · 1,063 images · 2,153 instances")

    st.markdown("---")
    st.subheader("🏷️ Defect Classes")
    for name, color in CLASS_COLORS.items():
        hex_color = "#%02x%02x%02x" % color
        st.markdown(
            f'<span class="class-chip"><span class="dot" style="background:{hex_color}"></span>{name}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.caption("Team: Aya (Data) · Heba (Model) · Saif (App)")

# ----------------------------------------------------------------------------
# Model loading (with error handling)
# ----------------------------------------------------------------------------
try:
    model = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Failed to load the model: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# Upload (supports multiple images)
# ----------------------------------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PCB image(s)",
    type=["jpg", "jpeg", "png", "bmp"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("⬆️ Upload one or more PCB images (JPG/PNG/BMP) to run defect detection.")
    st.stop()

valid_files = [f for f in uploaded_files if is_supported_image(f.name)]
invalid_count = len(uploaded_files) - len(valid_files)
if invalid_count:
    st.warning(f"⚠️ {invalid_count} file(s) skipped — unsupported type.")

all_rows = []
tab_labels = [f.name for f in valid_files]
tabs = st.tabs(tab_labels) if len(valid_files) > 1 else [st.container()]

for uploaded_file, tab in zip(valid_files, tabs):
    with tab:
        try:
            image = Image.open(uploaded_file)
        except Exception as e:
            st.error(f"Could not read '{uploaded_file.name}': {e}")
            continue

        with st.spinner(f"Running detection on {uploaded_file.name}..."):
            try:
                detections, annotated_image, elapsed_ms = run_inference(
                    model, image, conf=conf_threshold, iou=iou_threshold
                )
            except Exception as e:
                st.error(f"Inference failed on '{uploaded_file.name}': {e}")
                continue

        # ---- Summary metrics row ----
        m1, m2, m3 = st.columns(3)
        for col, (val, lbl) in zip(
            (m1, m2, m3),
            [
                (len(detections), "Defects found"),
                (f"{elapsed_ms:.0f} ms", "Inference time"),
                (f"{image.size[0]}×{image.size[1]}", "Image size"),
            ],
        ):
            col.markdown(
                f'<div class="metric-card"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(annotated_image, use_container_width=True)

            buf = io.BytesIO()
            annotated_image.save(buf, format="PNG")
            st.download_button(
                "⬇️ Download annotated image",
                data=buf.getvalue(),
                file_name=f"detected_{Path(uploaded_file.name).stem}.png",
                mime="image/png",
                use_container_width=True,
            )

        st.markdown("---")

        if len(detections) == 0:
            st.success("✅ No defects detected in this image.")
        else:
            st.subheader(f"⚠️ {len(detections)} Defect(s) Detected")
            table_data = [
                {
                    "Defect Type": d["class_name"],
                    "Confidence": f"{d['confidence']:.2%}",
                    "Location (x1, y1, x2, y2)": " , ".join(f"{v:.0f}" for v in d["box"]),
                }
                for d in detections
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

        for d in detections:
            all_rows.append(
                {
                    "image": uploaded_file.name,
                    "defect_type": d["class_name"],
                    "confidence": round(d["confidence"], 4),
                    "x1": round(d["box"][0], 1),
                    "y1": round(d["box"][1], 1),
                    "x2": round(d["box"][2], 1),
                    "y2": round(d["box"][3], 1),
                }
            )

# ----------------------------------------------------------------------------
# Batch summary (only shown for multi-image uploads)
# ----------------------------------------------------------------------------
if len(valid_files) > 1:
    st.markdown("## 📊 Batch Summary")
    if all_rows:
        df = pd.DataFrame(all_rows)
        s1, s2, s3 = st.columns(3)
        s1.metric("Images processed", len(valid_files))
        s2.metric("Total defects", len(df))
        s3.metric("Most common defect", df["defect_type"].mode().iloc[0])

        st.dataframe(
            df["defect_type"].value_counts().rename_axis("Defect Type").reset_index(name="Count"),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "⬇️ Download all detections (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="pcb_detections.csv",
            mime="text/csv",
        )
    else:
        st.success("✅ No defects detected across any of the uploaded images.")

st.markdown(
    '<div class="footer-note">PCB Defect Detection · YOLO11n · Team: Aya · Heba · Saif</div>',
    unsafe_allow_html=True,
)
