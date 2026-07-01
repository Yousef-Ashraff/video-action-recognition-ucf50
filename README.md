# Video Action Recognition — UCF50

A Gradio web app that classifies human actions in short video clips using a
Keras 3D-CNN model trained on the [UCF50](https://www.crcv.ucf.edu/data/UCF50.php)
action recognition dataset (50 action categories).

Upload a video, the app samples frames, runs them through the trained model,
and returns the top-3 predicted action classes with confidence scores.

## Demo

```
Upload Video (.avi or .mp4) → Predicted Action + Confidence
```

The UI is built with [Gradio](https://www.gradio.app/) (`gr.Interface`) and
exposes a single `gr.File` input and a `gr.Label` output showing the top 3
predicted classes.

## How it works

1. **Video conversion** — the uploaded file is transcoded to `.mp4`
   (H.264, `libx264`) with `ffmpeg` for consistent decoding.
2. **Frame extraction** — 20 frames are sampled at even intervals across the
   video with OpenCV, resized to 64×64, and normalized to `[0, 1]`. Short
   videos are padded by repeating the last frame.
3. **Inference** — the frame stack (`1, 20, 64, 64, 3`) is fed into the
   trained Keras model, which outputs a probability distribution over the
   50 UCF50 action classes.
4. **Result** — the top predictions are displayed with their confidence
   scores.

## Model

| | |
|---|---|
| File | `action_model.keras` (tracked with Git LFS, ~76 MB) |
| Frames per clip | 20 |
| Frame size | 64×64 (RGB) |
| Classes | 50 (UCF50) |
| Framework | TensorFlow / Keras |

<details>
<summary>Full list of the 50 action classes</summary>

BaseballPitch, Basketball, BenchPress, Biking, Billiards, BreastStroke,
CleanAndJerk, Diving, Drumming, Fencing, GolfSwing, HighJump, HorseRace,
HorseRiding, HulaHoop, JavelinThrow, JugglingBalls, JumpRope, JumpingJack,
Kayaking, Lunges, MilitaryParade, Mixing, Nunchucks, PizzaTossing,
PlayingGuitar, PlayingPiano, PlayingTabla, PlayingViolin, PoleVault,
PommelHorse, PullUps, Punch, PushUps, RockClimbingIndoor, RopeClimbing,
Rowing, SalsaSpin, SkateBoarding, Skiing, Skijet, SoccerJuggling, Swing,
TaiChi, TennisSwing, ThrowDiscus, TrampolineJumping, VolleyballSpiking,
WalkingWithDog, YoYo

</details>

## Project structure

```
.
├── app.py                # Gradio app: preprocessing + inference + UI
├── action_model.keras    # Trained model weights (Git LFS)
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Legacy static template (unused by the Gradio app)
└── .gitattributes         # Git LFS config for *.keras files
```

## Requirements

- Python 3.9+
- [`ffmpeg`](https://ffmpeg.org/) available on your `PATH` (used for video
  transcoding before frame extraction)
- [Git LFS](https://git-lfs.com/) to fetch the model file when cloning

## Setup

```bash
# Clone (Git LFS must be installed to download action_model.keras)
git lfs install
git clone https://github.com/Yousef-Ashraff/video-action-recognition-ucf50.git
cd video-action-recognition-ucf50

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Make sure `ffmpeg` is installed and on your `PATH`:

```bash
# Windows (winget)
winget install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Debian/Ubuntu
sudo apt install ffmpeg
```

## Usage

```bash
python app.py
```

This starts a local Gradio server (by default at `http://127.0.0.1:7860`).
Open it in your browser, upload a `.avi` or `.mp4` clip, and view the
predicted action.

## Notes & limitations

- Video-to-MP4 conversion writes temporary files to `/tmp`, which is
  Linux/macOS-specific; on Windows you may need to adjust `convert_to_mp4`
  in `app.py` to use a platform-appropriate temp directory (e.g.
  `tempfile.gettempdir()`).
- The model expects exactly 20 frames at 64×64 resolution — this must match
  the input shape used during training.

## License

No license has been specified for this project.
