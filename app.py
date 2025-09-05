#!/usr/bin/env python3
"""
Streamlit UI (revamped) for Traffic Light Detector

Clean, tabbed interface for images, videos, webcam snapshots, samples, and settings.
"""

# 1) Imports (OpenCV must be imported first to detect missing dependency early)
try:
    import cv2
    OPENCV_AVAILABLE = True
except Exception:
    cv2 = None
    OPENCV_AVAILABLE = False

import streamlit as st
import numpy as np
import tempfile
import os
from datetime import datetime
from typing import Optional

# Optional: live webcam via WebRTC
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration  # type: ignore
    import av  # type: ignore
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

# 2) Fail fast if OpenCV is missing
if not OPENCV_AVAILABLE:
    st.set_page_config(page_title="Traffic Light Detector", page_icon="🚦")
    st.error("OpenCV not found. Install: pip install opencv-python-headless")
    st.stop()

# 3) Import detector
try:
    from traffic_light_detector import AutoTrafficLightDetector
except Exception as e:
    st.set_page_config(page_title="Traffic Light Detector", page_icon="🚦")
    st.error(f"Failed to import detector: {e}")
    st.stop()

# 4) Page setup
st.set_page_config(page_title="Traffic Light Detector", page_icon="🚦", layout="wide")

st.markdown(
        """
        <style>
            :root { --brand:#22c55e; --brand2:#16a34a; --ink:#e5e7eb; --muted:#9aa0a6; }
            .title {text-align:center; font-size:2.6rem; font-weight:900; letter-spacing:.2px; margin:.35rem 0 .2rem;}
            .subtitle {text-align:center; color:var(--muted); margin-bottom:1rem;}
            .hero {
                border-radius:16px; padding:18px; margin:8px 0 16px; text-align:center;
                background: radial-gradient(1200px 400px at 20% -20%, rgba(34,197,94,0.18), rgba(59,130,246,0.12) 40%, transparent 70%),
                            linear-gradient(135deg, rgba(34,197,94,0.08), rgba(59,130,246,0.08));
                border: 1px solid rgba(255,255,255,0.06);
                backdrop-filter: blur(4px);
            }
            .pill {display:inline-block; padding:6px 12px; border-radius:999px; background:#0b1220; color:#a7f3d0; font-size:12px; border:1px solid rgba(255,255,255,0.06)}
            /* Buttons */
            .stButton>button {
                background: linear-gradient(180deg, var(--brand), var(--brand2));
                color:#02140a; font-weight:800; border:none; border-radius:12px;
                padding:.7rem 1.2rem; box-shadow: 0 8px 20px rgba(34,197,94,0.15);
                transition: transform .06s ease, box-shadow .2s ease, filter .2s ease;
            }
            .stButton>button:hover { transform: translateY(-1px); filter: brightness(1.05); box-shadow:0 12px 26px rgba(34,197,94,0.22); }
            .stButton>button:active { transform: translateY(0); filter: brightness(.98); }
            /* Tabs — catchy sticky glass bar with animated underline */
            div[role="tablist"] {
                position: sticky; top: 0; z-index: 5;
                padding: 6px; margin: 2px 0 12px;
                border-radius: 12px;
                background: linear-gradient(135deg, rgba(34,197,94,0.06), rgba(59,130,246,0.06));
                border: 1px solid rgba(255,255,255,0.06);
                backdrop-filter: blur(6px);
                box-shadow: 0 6px 18px rgba(0,0,0,0.18);
            }
            div[role="tablist"] button[role="tab"] {
                font-weight: 800;
                color: var(--muted);
                border-radius: 10px;
                padding: 8px 14px;
                position: relative;
                transition: color .15s ease, transform .06s ease;
            }
            div[role="tablist"] button[role="tab"]:hover { color: var(--ink); transform: translateY(-1px); }
            div[role="tablist"] button[role="tab"]::after {
                content: ""; position: absolute; left: 10px; right: 10px; bottom: -6px; height: 3px;
                background: linear-gradient(90deg, var(--brand), #60a5fa);
                border-radius: 999px; width: 0; opacity: .0; transition: all .18s ease;
            }
            div[role="tablist"] button[role="tab"]:hover::after { width: calc(100% - 20px); opacity: .65; }
            div[role="tablist"] button[role="tab"][aria-selected="true"] {
                color: #d1fae5;
                background: linear-gradient(180deg, rgba(34,197,94,0.22), rgba(22,163,74,0.14));
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.06), 0 6px 16px rgba(34,197,94,0.18);
            }
            div[role="tablist"] button[role="tab"][aria-selected="true"]::after { width: calc(100% - 20px); opacity: 1; }
            /* Metrics */
            [data-testid="stMetric"] { border:1px solid rgba(255,255,255,0.06); border-radius:12px; padding:8px 10px; background:rgba(255,255,255,0.02); }
            /* Images */
            img { border-radius:12px; }
            /* Sidebar & sliders */
            section[data-testid="stSidebar"] { border-right:1px solid rgba(255,255,255,0.06); }
            /* Progress */
            [data-testid="stProgressBar"] div div { background: linear-gradient(90deg, var(--brand), #60a5fa) !important; }
        </style>
        """,
        unsafe_allow_html=True,
)

st.markdown('<div class="hero"><div class="title">🚦 Traffic Light Detection</div><div class="subtitle">Simple, fast HSV-based detection for Red / Yellow / Green lights</div><span class="pill">Streamlit UI</span></div>', unsafe_allow_html=True)


# 5) Detector (cached)
@st.cache_resource(show_spinner=False)
def get_detector() -> AutoTrafficLightDetector:
    return AutoTrafficLightDetector()


detector = get_detector()


# 6) Utilities
def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else img


def draw_detections(base: np.ndarray, detections: list) -> np.ndarray:
    out = base.copy()
    for d in detections:
        x, y, w, h = d["box"]
        color = d.get("color", "?")
        conf = d.get("confidence", 0.0)
        if color == "Red":
            c = (0, 0, 255)
        elif color == "Yellow":
            c = (0, 200, 255)
        else:
            c = (0, 200, 0)
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
        cv2.rectangle(out, (x, y - 24), (x + w, y), c, -1)
        cv2.putText(out, f"{color} {conf:.2f}", (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def debug_masks(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        dyn = detector._calculate_dynamic_params(image)
        pre = detector._gray_world_white_balance(image)
        pre = detector._auto_gamma_correct(pre)
        blurred = cv2.GaussianBlur(pre, dyn["blur_kernel_size"], 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        bright_thresh = max(20, min(60, int(np.percentile(gray, 15))))
        _, bright_mask = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)
        if np.mean(s) < 15:
            combined = bright_mask
        else:
            _, sat_mask = cv2.threshold(s, 5, 255, cv2.THRESH_BINARY)
            combined = cv2.bitwise_and(bright_mask, sat_mask)
        panel = np.zeros_like(image)
        for color, ranges in detector.color_ranges.items():
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            for r in ranges:
                m = cv2.inRange(hsv, r["lower"], r["upper"])
                m = cv2.bitwise_and(m, combined)
                mask = cv2.bitwise_or(mask, m)
            k_small = np.ones((dyn["small_kernel_size"], dyn["small_kernel_size"]), np.uint8)
            k_close = np.ones((max(7, dyn["large_kernel_size"]), max(7, dyn["large_kernel_size"])), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_small)
            mask = cv2.medianBlur(mask, 3)
            if color == "Red":
                col = (0, 0, 255)
            elif color == "Yellow":
                col = (0, 255, 255)
            else:
                col = (0, 255, 0)
            colored = cv2.merge([
                mask // 2 if col[0] else np.zeros_like(mask),
                mask // 2 if col[1] else np.zeros_like(mask),
                mask // 2 if col[2] else np.zeros_like(mask),
            ])
            panel = cv2.add(panel, colored)
        return panel
    except Exception:
        return None


# Small helper to render ASCII labels in bold (Unicode mathematical bold)
def _to_bold(text: str) -> str:
    bold_map = {}
    # Build mapping lazily once
    if not hasattr(_to_bold, "_map"):
        m = {}
        A, a, zero = ord('A'), ord('a'), ord('0')
        # Mathematical Bold capital A starts at U+1D400, lowercase at U+1D41A, digits at U+1D7CE
        for i in range(26):
            m[chr(A + i)] = chr(0x1D400 + i)
            m[chr(a + i)] = chr(0x1D41A + i)
        for i in range(10):
            m[chr(zero + i)] = chr(0x1D7CE + i)
        _to_bold._map = m  # type: ignore[attr-defined]
    bold_map = getattr(_to_bold, "_map")  # type: ignore[attr-defined]
    return "".join(bold_map.get(ch, ch) for ch in text)


# Cached loader for sample images
@st.cache_data(show_spinner=False)
def load_sample_bgr(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(path)
        return img
    except Exception:
        return None


# 7) Sidebar settings
with st.sidebar:
    st.header("Settings")
    conf_thr = st.slider("Confidence threshold", 0.1, 1.0, 0.45, 0.05)
    show_dbg = st.toggle("Show debug masks", value=False, help="Visualize color segmentation")
    frame_skip = st.slider("Video frame skip", 1, 5, 2, 1, help="Process every Nth frame for speed")
    st.caption("Tip: Reduce frame rate or increase skip for faster video processing")

detector.confidence_threshold = conf_thr


# 8) Tabs
tab_img, tab_vid, tab_cam, tab_samples, tab_about = st.tabs([
    f"🖼️ {_to_bold('Image')}",
    f"📹 {_to_bold('Video')}",
    f"📷 {_to_bold('Webcam (live)')}",
    f"📁 {_to_bold('Samples')}",
    f"ℹ️ {_to_bold('About')}"
])


with tab_img:
    st.subheader("Upload an image")
    up = st.file_uploader("Image file", type=["jpg", "jpeg", "png", "bmp", "tiff"])
    if up is not None:
        data = np.asarray(bytearray(up.read()), dtype=np.uint8)
        img = cv2.imdecode(data, 1)
        if img is None:
            st.error("Failed to read image")
        else:
            st.image(bgr_to_rgb(img), channels="RGB", width='stretch', caption=up.name)
            if st.button("🚀 Detect", type="primary"):
                dets = detector.detect_traffic_lights(img)
                vis = draw_detections(img, dets)
                st.image(bgr_to_rgb(vis), channels="RGB", width='stretch', caption="Detections")
                # Stats
                rc = sum(1 for d in dets if d["color"] == "Red")
                yc = sum(1 for d in dets if d["color"] == "Yellow")
                gc = sum(1 for d in dets if d["color"] == "Green")
                col1, col2, col3 = st.columns(3)
                col1.metric("Red", rc)
                col2.metric("Yellow", yc)
                col3.metric("Green", gc)
                if show_dbg:
                    dbg = debug_masks(img)
                    if dbg is not None:
                        st.image(bgr_to_rgb(dbg), channels="RGB", width='stretch', caption="Debug masks")


with tab_vid:
    st.subheader("Upload a video")
    vup = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv", "wmv"])
    if vup is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{vup.name.split('.')[-1]}") as tf:
            tf.write(vup.read())
            vpath = tf.name
        st.success(f"Loaded: {os.path.basename(vup.name)}")
        run = st.button("▶️ Process video", type="primary")
        if run:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                st.error("Could not open video")
            else:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                st.caption(f"Frames: {total} | FPS: {fps:.1f}")
                ph = st.empty()
                pbar = st.progress(0)
                counts = {"Red": 0, "Yellow": 0, "Green": 0}
                i = 0
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    i += 1
                    if (i - 1) % frame_skip != 0:
                        continue
                    result = detector.process_frame(frame)
                    ph.image(bgr_to_rgb(result), channels="RGB", width='stretch')
                    # Count by raw detection on the same frame
                    for d in detector.detect_traffic_lights(frame):
                        if d["color"] in counts:
                            counts[d["color"]] += 1
                    if total > 0:
                        pbar.progress(min(1.0, i / total))
                cap.release()
                st.success("Video processing complete")
                c1, c2, c3 = st.columns(3)
                c1.metric("Red", counts["Red"]) ; c2.metric("Yellow", counts["Yellow"]) ; c3.metric("Green", counts["Green"])
        # Cleanup temp file when tab re-runs
        try:
            os.unlink(vpath)
        except Exception:
            pass


with tab_cam:
    st.subheader("Webcam (live)")
    if WEBRTC_AVAILABLE:
        st.caption("Live webcam in-browser via WebRTC. Works locally and in most browsers.")
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        def video_frame_callback(frame: "av.VideoFrame") -> "av.VideoFrame":
            img = frame.to_ndarray(format="bgr24")
            result = detector.process_frame(img)
            if show_dbg:
                # Optionally overlay a small count bar
                dets = detector.detect_traffic_lights(img)
                red = sum(1 for d in dets if d["color"] == "Red")
                yel = sum(1 for d in dets if d["color"] == "Yellow")
                grn = sum(1 for d in dets if d["color"] == "Green")
                cv2.rectangle(result, (0, 0), (220, 26), (0, 0, 0), -1)
                cv2.putText(result, f"R:{red} Y:{yel} G:{grn}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

        webrtc_streamer(
            key="traffic-light-webcam",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            rtc_configuration=rtc_config,
        )
    else:
        st.warning("Live webcam requires 'streamlit-webrtc'. Falling back to snapshot mode. Install with: pip install streamlit-webrtc av")
        snap = st.camera_input("Webcam snapshot")
        if snap is not None:
            file_bytes = np.asarray(bytearray(snap.getvalue()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            if img is None:
                st.error("Failed to decode snapshot")
            else:
                dets = detector.detect_traffic_lights(img)
                vis = draw_detections(img, dets)
                st.image(bgr_to_rgb(vis), channels="RGB", width='stretch', caption="Detections")
                if show_dbg:
                    dbg = debug_masks(img)
                    if dbg is not None:
                        st.image(bgr_to_rgb(dbg), channels="RGB", width='stretch', caption="Debug masks")


with tab_samples:
    st.subheader("Try sample images")
    st.caption("Click a card to run detection. Use 'Detect All' to run on the whole set.")

    samples = [
        ("Red Light", "sample_images/sample_red_light.jpg"),
        ("Yellow Light", "sample_images/sample_yellow_light.jpg"),
        ("Green Light", "sample_images/sample_green_light.jpg"),
        ("All Lights", "sample_images/sample_all_lights.jpg"),
        ("Multiple Lights", "sample_images/sample_multiple_lights.jpg"),
        ("Night Scene", "sample_images/sample_night_scene.jpg"),
        ("Challenging", "sample_images/sample_challenging_scene.jpg"),
    ]

    # Small CSS to make card look
    st.markdown(
        """
        <style>
          .card {background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:10px; margin-bottom:12px}
          .card h4 {margin:6px 0 8px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Detect all button
    col_head = st.container()
    with col_head:
        c1, c2 = st.columns([1, 3])
        with c1:
            run_all = st.button("Detect All", type="primary")
        with c2:
            st.empty()

    # Grid of cards
    grid_cols = st.columns(3, vertical_alignment="top")
    triggered_single = None
    for i, (label, path) in enumerate(samples):
        with grid_cols[i % 3]:
            if not os.path.exists(path):
                st.info(f"Missing: {path}")
                continue
            img = load_sample_bgr(path)
            if img is None:
                st.warning(f"Could not load {os.path.basename(path)}")
                continue
            with st.container(border=True):
                st.image(bgr_to_rgb(img), channels="RGB", width='stretch')
                st.markdown(f"**{label}**")
                if st.button(f"Detect • {label}"):
                    triggered_single = (label, path)

    st.markdown("---")
    st.subheader("Results")
    result_placeholder = st.empty()

    def run_and_show(img_bgr: np.ndarray, title: str):
        dets = detector.detect_traffic_lights(img_bgr)
        vis = draw_detections(img_bgr, dets)
        result_placeholder.image(bgr_to_rgb(vis), channels="RGB", width='stretch', caption=f"{title}")
        rc = sum(1 for d in dets if d["color"] == "Red")
        yc = sum(1 for d in dets if d["color"] == "Yellow")
        gc = sum(1 for d in dets if d["color"] == "Green")
        m1, m2, m3 = st.columns(3)
        m1.metric("Red", rc)
        m2.metric("Yellow", yc)
        m3.metric("Green", gc)
        if show_dbg:
            dbg = debug_masks(img_bgr)
            if dbg is not None:
                st.image(bgr_to_rgb(dbg), channels="RGB", width='stretch', caption="Debug masks")

    # Execute single
    if triggered_single is not None:
        title, p = triggered_single
        img = load_sample_bgr(p)
        if img is not None:
            run_and_show(img, f"{title} — detections")

    # Execute all
    if 'detect_all' not in st.session_state:
        st.session_state.detect_all = False
    if run_all:
        st.session_state.detect_all = True
    if st.session_state.detect_all:
        for label, path in samples:
            img = load_sample_bgr(path)
            if img is None:
                continue
            st.markdown(f"### {label}")
            run_and_show(img, f"{label} — detections")
        # reset flag
        st.session_state.detect_all = False


with tab_about:
    st.subheader("About")
    st.markdown(
        """
    - HSV-based color segmentation with dynamic preprocessing
    - Works with images, videos, and webcam (live via WebRTC) or snapshots
    - Set confidence threshold in the sidebar
        
        Run locally:
        1. pip install -r requirements.txt
    2. pip install streamlit-webrtc av  # for live webcam
    3. streamlit run app.py
        """
    )

