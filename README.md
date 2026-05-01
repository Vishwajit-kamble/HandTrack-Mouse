# 🖐️ HandTrack Mouse – Virtual Mouse using Hand Tracking

AirCursor is a real-time computer vision project that enables users to control their system cursor using hand gestures.
Built using Python, OpenCV, MediaPipe, and PyAutoGUI, it demonstrates gesture-based human-computer interaction without any physical mouse.

---

## 🚀 Features

* 🖐️ Real-time hand tracking using MediaPipe
* 🎯 Cursor movement using index finger
* 👌 Pinch gesture (thumb + index) for clicking
* 🎚️ Smooth cursor movement (reduces jitter)
* 🧱 Modular + phase-based implementation

---

## 🧠 Tech Stack

* Python
* OpenCV
* MediaPipe
* PyAutoGUI
* NumPy

---

## 🎯 How It Works

1. Captures webcam input using OpenCV
2. Detects hand landmarks (21 keypoints) using MediaPipe
3. Tracks index finger tip for cursor movement
4. Maps camera coordinates to screen resolution
5. Detects pinch gesture for mouse click
6. Applies smoothing for stable cursor control

---

## 📂 Project Structure

```id="k2r9bc"
virtual_mouse/
│
├── utils/
│   ├── hand_detector.py        # MediaPipe wrapper
│   └── mouse_controller.py     # Cursor + click logic
│
├── phase1_landmarks.py         # Detect and display landmarks
├── phase2_cursor.py            # Cursor movement
├── phase3_click.py             # Click + smoothing
│
├── virtual_mouse.py            # Final integrated app
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

### 1. Clone repository

```id="7k3h5u"
git clone https://github.com/your-username/AirCursor.git
cd AirCursor
```

### 2. Install dependencies

```id="2zq8ny"
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Final Version

```id="g8a4lo"
python virtual_mouse.py
```

### Phase-wise Learning (Optional)

```id="qvkjwb"
python phase1_landmarks.py
python phase2_cursor.py
python phase3_click.py
```

Press `ESC` to exit.

---

## ⚠️ Requirements

* Python 3.8 – 3.11
* Webcam
* Good lighting for accurate tracking

---

## 🎥 Demo

(Add demo video / GIF here)

---

## 🚀 Future Improvements

* Right-click gesture
* Scroll control
* Drag & drop
* Gesture shortcuts
* GUI control panel

---

## 👨‍💻 Author

**Vishwajit Kamble**

---

## ⭐ Support

If you found this useful, give it a star ⭐
