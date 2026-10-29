"""Streamlit interface for the educational DR classification prototype."""
import hashlib
import io
import re

import streamlit as st
from PIL import Image, UnidentifiedImageError

from app_core import gradcam, load_predictor, predict, quality_check
from metadata_store import delete_record, list_records, save_record, stats, update_note
from rag_core import guidance, retrieve

st.set_page_config(page_title="Diabetic Retinopathy Screening Demo", page_icon="👁️", layout="wide")
st.title("Diabetic Retinopathy Screening Demo")
st.error("EDUCATIONAL USE ONLY — This project is not a medical device and cannot diagnose or rule out diabetic retinopathy. Do not use its predictions or visualizations to make medical decisions. Consult a qualified eye-care professional for care.")

with st.expander("What the five grades mean", expanded=True):
    st.markdown("""
- **0 — No DR:** no diabetic retinopathy signs identified in the grading scheme.
- **1 — Mild:** mild non-proliferative changes, typically small microaneurysms.
- **2 — Moderate:** more retinal changes than mild disease, below the severe category.
- **3 — Severe:** severe non-proliferative changes in the standardized grading scale.
- **4 — Proliferative DR:** abnormal new blood vessel growth is present in the grading scheme.

These are simplified educational descriptions, not an assessment of an individual image.
""")


@st.cache_resource
def predictor():
    return load_predictor()


model, model_error = predictor()
if model is None:
    st.warning(f"Model unavailable: {model_error} No predictions will be generated until a trained checkpoint is available.")

tab_screen, tab_history = st.tabs(["Educational screening demo", "Demo history"])
with tab_screen:
    uploaded = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])
    demo_id = st.text_input("Synthetic/demo ID only", placeholder="demo-001")
    if uploaded:
        payload = uploaded.getvalue()
        try:
            image = Image.open(io.BytesIO(payload)).convert("RGB")
            st.image(image, caption="Uploaded image (not retained)", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("The uploaded file could not be decoded as an image.")
            image = None
        if image is not None:
            accepted, quality_message = quality_check(image)
            if not accepted:
                st.error(quality_message)
            elif "may be blurry" in quality_message or "brightness may limit" in quality_message:
                st.warning(quality_message)
            else:
                st.info(quality_message)
            if st.button("Analyze image", type="primary", disabled=(not accepted or model is None)):
                result = predict(model, image)
                if not result["ok"]:
                    st.error(f"Prediction failed: {result['message']}")
                else:
                    st.subheader("Educational model output")
                    left, right = st.columns(2)
                    left.metric("Predicted grade", f"{result['grade']} — {result['label']}")
                    right.metric("Prediction confidence", f"{result['confidence']:.1%}")
                    st.caption("Confidence is the model's softmax score, not calibrated clinical certainty. Grade ≥2 is a project convention, not a clinical threshold.")
                    st.bar_chart(result["probabilities"])
                    try:
                        heatmap, overlay = gradcam(model, image, result["grade"])
                        cam_col1, cam_col2 = st.columns(2)
                        cam_col1.image(heatmap, caption="Grad-CAM heatmap (experimental)", use_container_width=True)
                        cam_col2.image(overlay, caption="Grad-CAM overlay on model input", use_container_width=True)
                    except Exception as exc:
                        st.warning(f"Experimental Grad-CAM could not be generated: {exc}")
                    retrieval_error = None
                    try:
                        excerpts = retrieve(f"Educational diabetic retinopathy information related to grade {result['grade']} ({result['label']})")
                    except Exception:
                        excerpts, retrieval_error = [], "The local retrieval index is unavailable or unreadable."
                    text, guidance_status = guidance(result["grade"], excerpts)
                    st.subheader("Educational information")
                    st.write(text)
                    st.caption(guidance_status)
                    if retrieval_error:
                        st.warning(retrieval_error)
                    if excerpts:
                        st.markdown("**Retrieved source excerpts**")
                        for item in excerpts:
                            st.caption(f"{item['citation']} {item.get('document_title', item['source'])} ({item['source']}), page {item['page']} · retrieval score {item['score']:.3f}")
                            st.write(item["text"])
                    valid_demo = bool(re.fullmatch(r"demo-[A-Za-z0-9_-]{1,48}", demo_id.strip()))
                    if valid_demo:
                        ok, status = save_record({"demo_id": demo_id.strip(), "image_sha256": hashlib.sha256(payload).hexdigest(),
                                                  "grade": result["grade"], "label": result["label"],
                                                  "confidence": result["confidence"], "referable_demo_group": result["referable_demo_group"],
                                                  "guidance": text, "sources": [{"source": e["source"], "page": e["page"], "score": e["score"]} for e in excerpts]})
                        st.caption(status)
                    elif demo_id.strip():
                        st.info("History storage accepts synthetic IDs in the form demo-001. No record was saved.")

with tab_history:
    history_id = st.text_input("Synthetic/demo ID", key="history_id", placeholder="demo-001")
    if st.button("Load history"):
        if not re.fullmatch(r"demo-[A-Za-z0-9_-]{1,48}", history_id.strip()):
            st.info("Enter a synthetic ID in the form demo-001.")
        else:
            rows, error = list_records(history_id.strip())
            st.session_state["history_records"] = rows
            st.session_state["history_error"] = error
            st.session_state["history_demo_id"] = history_id.strip() if error is None else None

    notice = st.session_state.pop("history_notice", None)
    if notice:
        st.success(notice)
    history_error = st.session_state.get("history_error")
    if history_error:
        st.error(history_error)

    loaded_demo_id = st.session_state.get("history_demo_id")
    if loaded_demo_id:
        records = st.session_state.get("history_records", [])
        if records:
            st.subheader("Screening records")
            st.dataframe(records, use_container_width=True)
        elif not history_error:
            st.info("No screening records were found for this demo ID.")

        st.subheader("Grade statistics")
        stats_rows, stats_error = stats(loaded_demo_id)
        if stats_error:
            st.error(stats_error)
        elif stats_rows:
            st.dataframe(stats_rows, use_container_width=True)
            st.bar_chart({str(row["grade"]): row["count"] for row in stats_rows})
        else:
            st.info("No grade statistics are available for this demo ID.")

        record_ids = [row["id"] for row in records]
        if record_ids:
            record_by_id = {row["id"]: row for row in records}
            if st.session_state.get("history_selected_id") not in record_ids:
                st.session_state["history_selected_id"] = record_ids[0]
            selected_id = st.selectbox(
                "Select a record",
                record_ids,
                key="history_selected_id",
                format_func=lambda value: f"Grade {record_by_id[value].get('grade', 'unknown')} · {record_by_id[value].get('created_at', '')}",
            )
            selected_record = record_by_id[selected_id]
            with st.form(f"review_note_form_{selected_id}"):
                reviewer_note = st.text_area(
                    "Reviewer note",
                    value=selected_record.get("reviewer_note", ""),
                    max_chars=500,
                    key=f"reviewer_note_{selected_id}",
                )
                save_note = st.form_submit_button("Save reviewer note")
            if save_note:
                ok, message = update_note(selected_id, reviewer_note)
                if ok:
                    refreshed, error = list_records(loaded_demo_id)
                    st.session_state["history_records"] = refreshed
                    st.session_state["history_error"] = error
                    st.success(message)
                else:
                    st.error(message)

            confirm_delete = st.checkbox(
                "I confirm that I want to delete this screening record.",
                key=f"confirm_delete_{selected_id}",
            )
            if st.button("Delete selected record", disabled=not confirm_delete):
                ok, message = delete_record(selected_id)
                if ok:
                    refreshed, error = list_records(loaded_demo_id)
                    st.session_state["history_records"] = refreshed
                    st.session_state["history_error"] = error
                    st.session_state["history_notice"] = message
                    st.rerun()
                else:
                    st.error(message)

st.caption("Predictions, image-quality checks, and Grad-CAM heatmaps can be wrong or misleading. Educational use only; seek advice from a qualified professional.")
