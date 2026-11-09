# Demo recording

This demo is a condensed replay of real stdout from the unchanged `mini_rl_teacher.py`, using its default 1,000 training iterations. It is not a desktop screen recording or a simulated training animation.

- [MP4](mini-rl-teacher.mp4): 1440 × 1000, H.264, 25 fps, approximately 43 seconds.
- [GIF preview](preview.gif): a smaller inline preview for the repository README.
- [Poster](poster.png): the final printed training frame.
- [Terminal transcript](terminal.txt): all captured stdout, with ANSI control sequences removed.
- [Run metadata](run.json): source/transcript SHA-256, environment versions, and final evaluation output.

All 101 printed training frames are replayed in sequence, including the unsuccessful outcome. Pretraining is held for 4 seconds, each training frame for 0.32 seconds, and final evaluation for 7 seconds. These presentation timings do not represent execution times. The original script's fixed success statement is retained in the transcript; measured evaluation, not that statement, determines success.

The recorded run used Python 3.11 and NumPy 2.4.6. Final deterministic evaluation over 50 episodes reported mean endpoint `(6.00, 4.78)`, mean endpoint distance `4.10`, mean true return `-91.79`, and `0%` within radius `0.5`. This run did not converge.

## Reproduce

From the repository root, using an activated virtual environment:

```bash
python -m pip install numpy pillow
python scripts/record_demo.py
```

Install `ffmpeg` separately and make it available on `PATH`. The recorder looks for a Chinese-capable font at common macOS, Linux, and Windows locations. Supply `--font /path/to/font.ttf` if needed; `.otf` and `.ttc` files are also supported. The font is used locally and is not bundled in the repository.

The recorder runs the training script as a subprocess without changing its source or parameters, captures its stdout, then renders the output to MP4 and GIF with Pillow and ffmpeg. Generated files replace those in `docs/demo/`; review the new metrics before updating the descriptions in this file and the root README.

To regenerate media from the included transcript without retraining:

```bash
python scripts/record_demo.py --replay docs/demo/terminal.txt --output /tmp/mini-rl-teacher-replay
```

Replay metadata records the current source hash and rendering environment; preserve the original `run.json` when replaying a recording from another environment. The parser expects the default 1,000-iteration run and the current terminal output format.
