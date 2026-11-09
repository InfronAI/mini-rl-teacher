# Demo recording

**English** | [简体中文](README.zh-CN.md)

These demos replay real stdout from the unchanged `mini_rl_teacher.py`, using its default 1,000 training iterations, 60-step episodes, and 32 episodes per iteration. They are not desktop screen recordings or simulated training animations. Both languages use the same recorded run, measurements, grid positions, and playback timing.

| Asset | English | 简体中文 |
| --- | --- | --- |
| MP4 video | [Watch / download](mini-rl-teacher.en.mp4) | [观看 / 下载](mini-rl-teacher.zh-CN.mp4) |
| GIF preview | [Preview](preview.en.gif) | [预览](preview.zh-CN.gif) |
| Poster | [Poster](poster.en.png) | [封面](poster.zh-CN.png) |
| Transcript | [Translated output](terminal.en.txt) | [Original output](terminal.txt) |
| Rendering metadata | [English](media.en.json) | [中文](media.zh-CN.json) |

Both MP4s are 1440 × 1000, H.264, 25 fps, and 43.32 seconds long. English translates the terminal text as well as the presentation labels. Chinese retains the original terminal output and uses Chinese stage titles and footers. The training script itself remains unchanged and still prints Chinese output.

All 101 printed training frames are replayed in sequence, including the 50/50 successful final evaluation. Pretraining is held for 4 seconds, each training frame for 0.32 seconds, and final evaluation for 7 seconds. These presentation timings do not represent execution times. The script's fixed success statement is retained (and translated in English); the evaluation heading explains that measured metrics determine success.

The recorded run used Python 3.11 and NumPy 2.4.6. Final deterministic evaluation over 50 episodes reported mean endpoint `(2.76, 1.81)`, mean endpoint distance `0.31`, mean true return `-35.60`, and `100%` within radius `0.5`. All 50 evaluation episodes passed the endpoint test. The evaluation does not test zero velocity or continued residence at the target.

[run.json](run.json) preserves the original recording's source/transcript SHA-256, environment versions, and evaluation output. Each language's rendering metadata references the same original transcript hash and records its own media hashes. The English transcript is a presentation translation, not stdout from a separate training run.

## Reproduce

From the repository root, using an activated virtual environment:

```bash
python -m pip install numpy pillow
python scripts/record_demo.py --language both
```

Install `ffmpeg` separately and make it available on `PATH`. The recorder looks for a Chinese-capable font at common macOS, Linux, and Windows locations. Supply `--font /path/to/font.ttf` if needed; `.otf` and `.ttc` files are also supported. The font is used locally and is not bundled in the repository.

The recorder runs fresh training once, captures stdout, and renders both language versions with Pillow and ffmpeg. The training script saves `reward_model.npz` in the repository root after pretraining and overwrites it after final evaluation. The recorder does not pass `--load`; an existing reward-model file does not skip pretraining. `--language` accepts `en`, `zh-CN`, or `both` (the default). Generated files replace the corresponding files in `docs/demo/`; update the measured results and source hash in both READMEs when recording a new code version.

To regenerate both versions from the included transcript without retraining:

```bash
python scripts/record_demo.py --replay docs/demo/terminal.txt --language both --output /tmp/mini-rl-teacher-replay
```

Replay expects the original Chinese transcript, not `terminal.en.txt`. If an adjacent `run.json` exists, its transcript hash is verified and its recording metadata is preserved. Without it, unavailable recording provenance is left unknown. Rendering environment details are saved separately in `media.en.json` and `media.zh-CN.json`. The parser expects the default 1,000-iteration run and the current output format; untranslated Chinese text in English rendering raises an error instead of silently mixing languages.
