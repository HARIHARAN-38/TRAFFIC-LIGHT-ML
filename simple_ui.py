#!/usr/bin/env python3
"""
Simple Traffic Light Detector - Minimal Tkinter UI

This UI avoids OpenCV GUI windows and works well on Windows by using Tkinter
for all interactions and display. It supports:
- Opening and processing a single image
- Playing and processing a video file
- Starting the webcam and processing frames live

Press Stop to end video/webcam playback. Use Save Frame to save the
currently displayed annotated image.
"""

import os
import sys
import time
import platform
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Import detector
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(str(script_dir))
try:
    from traffic_light_detector import AutoTrafficLightDetector
except ImportError:
    print("Error: Could not import AutoTrafficLightDetector from traffic_light_detector.py")
    sys.exit(1)

APP_TITLE = "Traffic Light Detector"
IS_WINDOWS = platform.system() == "Windows"


class SimpleTkUI:
    """A simple Tkinter UI for the traffic light detector."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("980x680")
        self.root.minsize(720, 480)

        # Detector instance
        self.detector = AutoTrafficLightDetector()

        # Video state
        self.cap = None
        self.running = False
        self.current_source = None  # "webcam", "video", "image"
        self.last_frame = None  # Numpy BGR annotated
        self.photo_image = None  # Keep reference to avoid GC

        # UI vars
        self.save_each_frame_var = tk.BooleanVar(value=False)

        self._build_ui()
        self._bind_events()

    # ---------------- UI Construction ----------------
    def _build_ui(self):
        # Top controls
        top = tk.Frame(self.root, padx=10, pady=8)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="Open Image", width=14, command=self.on_open_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Open Video", width=14, command=self.on_open_video).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Start Webcam", width=14, command=self.on_start_webcam).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Stop", width=10, command=self.on_stop).pack(side=tk.LEFT, padx=12)
        tk.Button(top, text="Save Frame", width=12, command=self.on_save_frame).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(top, text="Auto-save frames", variable=self.save_each_frame_var).pack(side=tk.LEFT, padx=10)
        tk.Button(top, text="Quit", width=10, command=self.on_quit).pack(side=tk.RIGHT)

        # Display area
        self.display = tk.Label(self.root, bg="#111")
        self.display.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Status bar
        self.status = tk.Label(self.root, text="Ready", anchor="w", bd=1, relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_quit)

    # ---------------- Helpers ----------------
    def _set_status(self, text: str):
        self.status.config(text=text)
        self.status.update_idletasks()

    def _bgr_to_photo(self, bgr: np.ndarray, max_w=1280, max_h=860) -> ImageTk.PhotoImage:
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # Fit to label size while keeping aspect ratio
        lbl_w = max(self.display.winfo_width(), 640)
        lbl_h = max(self.display.winfo_height(), 360)
        target_w = min(lbl_w, max_w)
        target_h = min(lbl_h, max_h)
        scale = min(target_w / w, target_h / h)
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img = Image.fromarray(resized)
        return ImageTk.PhotoImage(image=img)

    def _show_frame(self, frame_bgr: np.ndarray):
        self.photo_image = self._bgr_to_photo(frame_bgr)
        if self.photo_image is not None:
            self.display.config(image=self.photo_image)
            self.display.update_idletasks()

    def _stop_capture(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.running = False

    # ---------------- Button Callbacks ----------------
    def on_open_image(self):
        self.on_stop()
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._set_status(f"Processing image: {Path(path).name}")
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror(APP_TITLE, "Failed to load image.")
            self._set_status("Ready")
            return
        annotated = self.detector.process_single_image(img)
        self.last_frame = annotated
        self._show_frame(annotated)
        self.current_source = "image"
        self._set_status("Image processed")

    def on_open_video(self):
        self.on_stop()
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self._start_capture(path, source_name=Path(path).name)

    def on_start_webcam(self):
        self.on_stop()
        self._start_capture(0, source_name="Webcam")

    def on_stop(self):
        self._stop_capture()
        self._set_status("Stopped")

    def on_quit(self):
        self.on_stop()
        self.root.after(50, self.root.destroy)

    def on_save_frame(self):
        if self.last_frame is None:
            messagebox.showinfo(APP_TITLE, "Nothing to save yet.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = script_dir / f"traffic_light_frame_{ts}.jpg"
        try:
            cv2.imwrite(str(out_path), self.last_frame)
            self._set_status(f"Saved: {out_path.name}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"Failed to save image: {e}")

    # ---------------- Video/Webcam Loop ----------------
    def _start_capture(self, source, source_name: str):
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open source: {source_name}")
            self.running = True
            self.current_source = "webcam" if source == 0 else "video"
            self._set_status(f"Running: {source_name} (press Stop to end)")
            self._update_stream()
        except Exception as e:
            self._stop_capture()
            messagebox.showerror(APP_TITLE, f"Failed to start: {e}")
            self._set_status("Ready")

    def _update_stream(self):
        if not self.running or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self._set_status("End of stream")
            self.on_stop()
            return

        # Process and show
        annotated = self.detector.process_frame(frame)
        self.last_frame = annotated
        self._show_frame(annotated)

        # Optional auto-save
        if self.save_each_frame_var.get():
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                out_path = script_dir / f"frame_{ts}.jpg"
                cv2.imwrite(str(out_path), annotated)
            except Exception:
                pass

        # Aim for ~30 FPS UI updates if possible
        self.root.after(10, self._update_stream)


def main():
    # Basic notice about OpenCV build
    print(f"Running on {platform.system()} with OpenCV {cv2.__version__}")
    if "headless" in cv2.getBuildInformation().lower():
        print("OpenCV headless build detected (no native GUI) - Tkinter UI will be used.")

    root = tk.Tk()
    app = SimpleTkUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
