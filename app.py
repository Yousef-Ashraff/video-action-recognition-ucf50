import gradio as gr
import numpy as np
import cv2
import subprocess
import uuid
import tensorflow as tf
from tensorflow.keras.models import load_model

# ================= CONFIG =================
num_frames = 20
for_pretrain = False

n_h, n_w = 64, 64              # CHANGE to your trained size
MODEL_PATH = "action_model.keras"

CLASS_NAMES = [
    "BaseballPitch", "Basketball", "BenchPress", "Biking", "Billiards",
    "BreastStroke", "CleanAndJerk", "Diving", "Drumming", "Fencing",
    "GolfSwing", "HighJump", "HorseRace", "HorseRiding", "HulaHoop",
    "JavelinThrow", "JugglingBalls", "JumpRope", "JumpingJack", "Kayaking",
    "Lunges", "MilitaryParade", "Mixing", "Nunchucks", "PizzaTossing",
    "PlayingGuitar", "PlayingPiano", "PlayingTabla", "PlayingViolin", "PoleVault",
    "PommelHorse", "PullUps", "Punch", "PushUps", "RockClimbingIndoor",
    "RopeClimbing", "Rowing", "SalsaSpin", "SkateBoarding", "Skiing",
    "Skijet", "SoccerJuggling", "Swing", "TaiChi", "TennisSwing",
    "ThrowDiscus", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog", "YoYo"
]

# ================= LOAD MODEL ONCE =================
model = load_model(MODEL_PATH)

# ================= VIDEO CONVERSION =================
def convert_to_mp4(video_path):
    mp4_path = f"/tmp/{uuid.uuid4().hex}.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "faststart",
        mp4_path
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    return mp4_path

# ================= FRAME EXTRACTION =================
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame = cv2.resize(frame, (n_w, n_h))
            frame = frame / 255.0
            frames.append(frame)

        count += 1

    cap.release()

    # Pad short videos
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames, dtype=np.float32)

# ================= INFERENCE =================
def predict_video(video_file):
    video_path = video_file.name   # gr.File gives a temp file

    mp4_path = convert_to_mp4(video_path)
    frames = extract_frames(mp4_path)

    frames = np.expand_dims(frames, axis=0)  # (1, T, H, W, C)
    preds = model.predict(frames)[0]

    class_id = int(np.argmax(preds))
    confidence = float(preds[class_id])

    return {
        CLASS_NAMES[class_id]: confidence
    }

# ================= GRADIO UI =================
demo = gr.Interface(
    fn=predict_video,
    inputs=gr.File(label="Upload Video (.avi or .mp4)"),
    outputs=gr.Label(num_top_classes=3),
    title="Video Classification",
    description="Upload a video file and get the predicted action class."
)

if __name__ == "__main__":
    demo.launch()
