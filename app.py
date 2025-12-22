from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# UCF50 Class Labels (Alphabetical Order)
CLASS_LABELS = [
    'BaseballPitch', 'Basketball', 'BenchPress', 'Biking', 'Billiards',
    'BreastStroke', 'CleanAndJerk', 'Diving', 'Drumming', 'Fencing',
    'GolfSwing', 'HighJump', 'HorseRace', 'HorseRiding', 'HulaHoop',
    'JavelinThrow', 'JugglingBalls', 'JumpRope', 'JumpingJack', 'Kayaking',
    'Lunges', 'MilitaryParade', 'Mixing', 'Nunchucks', 'PizzaTossing',
    'PlayingGuitar', 'PlayingPiano', 'PlayingTabla', 'PlayingViolin', 'PoleVault',
    'PommelHorse', 'PullUps', 'Punch', 'PushUps', 'RockClimbingIndoor',
    'RopeClimbing', 'Rowing', 'SalsaSpin', 'SkateBoarding', 'Skiing',
    'Skijet', 'SoccerJuggling', 'Swing', 'TaiChi', 'TennisSwing',
    'ThrowDiscus', 'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog', 'YoYo'
]

# Model Configuration
NUM_FRAMES = 20
IMG_HEIGHT = 64
IMG_WIDTH = 64

# Load Model
print("Loading model...")
try:
    model = tf.keras.models.load_model('action_model.keras')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None


def extract_frames(video_path, num_frames=NUM_FRAMES, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    """Extract frames from video using same preprocessing as training."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)

    count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frame = cv2.resize(frame, (img_width, img_height))
            frame = frame / 255.0
            frames.append(frame)
        count += 1

    cap.release()

    # Pad if video too short
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((img_height, img_width, 3)))

    return np.array(frames, dtype=np.float32)


@app.route('/')
def index():
    return render_template('index.html', classes=CLASS_LABELS)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'success': False, 'error': 'Model not loaded.'}), 500

    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file uploaded.'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'success': False, 'error': 'No video file selected.'}), 400

    try:
        filename = secure_filename(video_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        video_file.save(temp_path)

        frames = extract_frames(temp_path)
        frames_batch = np.expand_dims(frames, axis=0)

        predictions = model.predict(frames_batch, verbose=0)
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]

        results = [{'class': CLASS_LABELS[idx], 'confidence': float(predictions[0][idx] * 100)} for idx in top_5_indices]

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return jsonify({
            'success': True,
            'predictions': results,
            'top_prediction': results[0]['class'],
            'top_confidence': results[0]['confidence']
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
