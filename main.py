import importlib.util
import os
import sys
import shutil
import tempfile
import cv2
import streamlit as st

st.set_page_config(page_title="Construction PPE Detection", layout="wide", page_icon="")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "code")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_VIDEO_PATH = os.path.join(CODE_DIR, "test_video.mp4")

def load_code_module(module_name, filename):
    file_path = os.path.join(CODE_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

model_eval = load_code_module("model_evaluation_testing", "1.model_evaluation_testing.py")
model_video = load_code_module("model_pipeline_video", "2.model_pipeline_video.py")

@st.cache_resource
def get_model():
    return model_video.load_pipeline_model()

model = get_model()

st.sidebar.title("PPE Safety System")
app_mode = st.sidebar.radio(
    "Navigation",
    [
        "Video Pipeline & Violation Detection",
        "Safety Violations & Snapshots",
        "Model Evaluation & Testing"
    ]
)

conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.15, step=0.05)

if app_mode == "Video Pipeline & Violation Detection":
    st.title("Construction Site PPE Video Pipeline")
    st.markdown("Real-time safety monitoring and automated 10-second violation clip evidence recording.")

    video_choice = st.sidebar.radio("Video Source", ["Default Sample (test_video.mp4)", "Upload Custom Video"])
    
    video_source_path = None
    temp_file_path = None

    if video_choice == "Default Sample (test_video.mp4)":
        if os.path.exists(DEFAULT_VIDEO_PATH):
            video_source_path = DEFAULT_VIDEO_PATH
        else:
            st.error(f"Sample video not found at: {DEFAULT_VIDEO_PATH}")
    else:
        uploaded_video = st.sidebar.file_uploader("Upload MP4 / AVI Footage", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()
            temp_file_path = tfile.name
            video_source_path = temp_file_path

    if video_source_path:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Source Footage")
            st.video(video_source_path)

        with col2:
            st.markdown("### Live Pipeline Analysis")
            status_box = st.empty()
            live_feed = st.empty()

        if st.sidebar.button("Run Pipeline", type="primary"):
            status_box.info("Pipeline processing video frames...")
            
            def on_frame(annotated_frame, frame_idx, current_violations, violation_triggered, evidence_path):
                rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                live_feed.image(rgb, channels="RGB", use_container_width=True)

                if current_violations:
                    status_box.error(f"Safety Violation: {', '.join(set(current_violations))}")
                elif violation_triggered:
                    status_box.warning("Recording 10s Violation Evidence Clip...")
                else:
                    status_box.success("Worker PPE Compliant")
                
                return False

            result = model_video.run_video_pipeline(
                video_source=video_source_path,
                model=model,
                conf_threshold=conf_threshold,
                output_directory=OUTPUT_DIR,
                frame_callback=on_frame
            )

            status_box.success(f"Processing Complete! Total frames analyzed: {result['total_frames']}.")
            
            evidence_clip = result.get("evidence_path")
            if not evidence_clip:
                possible_evidence = os.path.join(OUTPUT_DIR, "violation_evidence.mp4")
                if os.path.exists(possible_evidence):
                    evidence_clip = possible_evidence

            if evidence_clip and os.path.exists(evidence_clip):
                st.markdown("---")
                st.subheader("Captured Violation Evidence Clip")
                st.markdown(f"Saved 10-second clip with pre/post bounding boxes to: `{evidence_clip}`")
                st.video(evidence_clip)

            snapshots = result.get("snapshots", [])
            if snapshots:
                st.markdown("---")
                st.subheader("Flagged Violation Snapshots")
                st.markdown(f"Captured {len(snapshots)} violation snapshots. View full compartments in the **Safety Violations & Snapshots** page.")
                cols = st.columns(min(len(snapshots), 4))
                for idx, snap in enumerate(snapshots[:4]):
                    with cols[idx % 4]:
                        st.image(snap["path"], caption=f"Removed: {snap['class']} (Frame {snap['frame']})", use_container_width=True)
    else:
        st.info("Please select or upload a video from the sidebar to begin processing.")

elif app_mode == "Safety Violations & Snapshots":
    st.title("Safety Violations & Incident Snapshots")
    st.markdown("Flagged evidence showing workers who have removed required PPE gear, sorted into compartments.")

    violations_by_class = model_video.get_saved_violations(output_directory=OUTPUT_DIR)

    if not violations_by_class:
        st.info("No safety violations currently flagged. Run the video pipeline to detect workers removing required PPE.")
    else:
        total_snapshots = sum(len(imgs) for imgs in violations_by_class.values())
        
        m_col1, m_col2, m_col3 = st.columns([2, 2, 2])
        with m_col1:
            st.metric("Total Flagged Snapshots", total_snapshots)
        with m_col2:
            st.metric("Violation Categories", len(violations_by_class))
        with m_col3:
            if st.button("Clear All Violation Snapshots"):
                violations_dir = os.path.join(OUTPUT_DIR, "violations")
                if os.path.exists(violations_dir):
                    shutil.rmtree(violations_dir)
                st.rerun()

        class_keys = list(violations_by_class.keys())
        tab_titles = ["All Flagged Incidents"] + [k.replace('_', ' ').title() for k in class_keys]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            st.markdown("#### All Flagged Incidents")
            all_snapshots = []
            for v_class, img_paths in violations_by_class.items():
                for p in img_paths:
                    all_snapshots.append((v_class, p))

            cols = st.columns(3)
            for idx, (v_class, img_p) in enumerate(all_snapshots):
                with cols[idx % 3]:
                    st.image(img_p, caption=f"Violation: {v_class} | {os.path.basename(img_p)}", use_container_width=True)

        for i, v_class in enumerate(class_keys):
            with tabs[i + 1]:
                st.markdown(f"#### Compartment: {v_class.replace('_', ' ').title()}")
                img_paths = violations_by_class[v_class]
                cols = st.columns(3)
                for idx, img_p in enumerate(img_paths):
                    with cols[idx % 3]:
                        st.image(img_p, caption=f"Removed: {v_class} | {os.path.basename(img_p)}", use_container_width=True)

elif app_mode == "Model Evaluation & Testing":
    st.title("Model Evaluation & Dataset Metrics")
    st.markdown("Dataset class distribution, imbalance analysis, and model training metrics.")

    metrics = model_eval.get_available_metrics(output_dir=OUTPUT_DIR)

    st.markdown("### 1. Dataset Class Imbalance")
    class_balance_chart = metrics.get("class_balance")
    if class_balance_chart and os.path.exists(class_balance_chart):
        st.image(class_balance_chart, caption="Class instance counts in Construction PPE dataset", use_container_width=True)
    else:
        st.warning("Class balance chart not found. Click 'Generate / Refresh Class Balance Chart' above.")

    st.markdown("### 2. Training Performance & Validation Results")
    eval_tabs = st.tabs(["Results Overview", "Confusion Matrix", "F1 / Precision-Recall Curves"])

    with eval_tabs[0]:
        if "results" in metrics and os.path.exists(metrics["results"]):
            st.image(metrics["results"], caption="Training & Validation Losses and mAP Curves", use_container_width=True)
        else:
            st.info("Training results chart not found in output directory.")

    with eval_tabs[1]:
        cm_col1, cm_col2 = st.columns(2)
        with cm_col1:
            if "confusion_matrix" in metrics and os.path.exists(metrics["confusion_matrix"]):
                st.image(metrics["confusion_matrix"], caption="Confusion Matrix", use_container_width=True)
        with cm_col2:
            if "confusion_matrix_normalized" in metrics and os.path.exists(metrics["confusion_matrix_normalized"]):
                st.image(metrics["confusion_matrix_normalized"], caption="Normalized Confusion Matrix", use_container_width=True)

    with eval_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            if "BoxF1_curve" in metrics and os.path.exists(metrics["BoxF1_curve"]):
                st.image(metrics["BoxF1_curve"], caption="F1-Confidence Curve", use_container_width=True)
        with c2:
            if "BoxPR_curve" in metrics and os.path.exists(metrics["BoxPR_curve"]):
                st.image(metrics["BoxPR_curve"], caption="Precision-Recall Curve", use_container_width=True)

    violations_by_class = model_video.get_saved_violations(output_directory=OUTPUT_DIR)
    if violations_by_class:
        st.markdown("---")
        st.markdown("### 3. Video Pipeline Flagged Violations Summary")
        v_summary = {k: len(v) for k, v in violations_by_class.items()}
        st.write("Current flagged violations captured from processed video:", v_summary)