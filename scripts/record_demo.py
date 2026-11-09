#!/usr/bin/env python3
"""Capture the unchanged mini script's output and render a condensed terminal replay.

Requires NumPy (training), Pillow (rendering), ffmpeg, and a Chinese-capable font.
Use --replay PATH to render a previously captured UTF-8 terminal log instead.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
WIDTH, HEIGHT, FPS = 1440, 1000, 25


def find_font(explicit):
    candidates = [explicit] if explicit else [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    raise SystemExit("Pass --font with a Chinese-capable TTF/OTF/TTC font path.")


def parse_output(raw):
    # Remove carriage returns before decoding lines, preserving original line breaks.
    clean = ANSI.sub("", raw.replace("\r", ""))
    lines = clean.splitlines()
    starts = [i for i, line in enumerate(lines)
              if line.startswith("mini_rl_teacher · 2D")]
    if not starts:
        raise ValueError("No training frames found in the captured output")
    frames = []
    for start in starts:
        end = next(i for i in range(start, len(lines))
                   if "x[-6 .. +6]" in lines[i])
        frames.append(lines[start:end + 1])
    final_start = next(i for i, line in enumerate(lines) if "[最终评估" in line)
    if "iter 1000 / 1000" not in frames[-1][0]:
        raise ValueError("Expected a complete, default 1000-iteration run")
    return clean, lines[:starts[0]], frames, lines[final_start:]


def render(lines, stage, font, progress=None):
    im = Image.new("RGB", (WIDTH, HEIGHT), "#0b1020")
    d = ImageDraw.Draw(im)
    title_font = ImageFont.truetype(font, 32)
    text_font = ImageFont.truetype(font, 22)
    small_font = ImageFont.truetype(font, 19)
    d.text((48, 26), "MINI RL TEACHER", font=title_font, fill="#edf3ff")
    d.text((48, 70), stage, font=small_font, fill="#a5b5d5")
    d.rounded_rectangle((28, 112, WIDTH - 28, 925), 16, fill="#141c2e", outline="#34415c")
    d.text((52, 125), "$ python mini_rl_teacher.py", font=small_font, fill="#8cdda9")
    for row, line in enumerate(lines):
        x, y = 52, 165 + row * 28
        if y > 895:
            raise ValueError("Terminal output would be cropped")
        for ch in line:
            cells = 2 if unicodedata.east_asian_width(ch) in ("F", "W") else 1
            color = {"·": "#65728b", "▼": "#ff7b86", "●": "#7be0aa", "★": "#7be0aa"}.get(ch, "#e1e8f5")
            if x + cells * 14 > WIDTH - 48:
                raise ValueError("Terminal line is too wide to render")
            d.text((x, y), ch, font=text_font, fill=color)
            x += cells * 14
    d.text((48, 948), "Actual stdout replay | Uniform frame timing; not real-time | Default seeds and parameters",
           font=small_font, fill="#9eacc7")
    if progress is not None:
        d.rectangle((28, 916, 28 + (WIDTH - 56) * progress, 920), fill="#7be0aa")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay", type=Path, help="Existing raw or plain UTF-8 stdout log")
    ap.add_argument("--font", help="Chinese-capable font path")
    ap.add_argument("--output", type=Path, default=ROOT / "docs/demo")
    args = ap.parse_args()
    font = find_font(args.font)
    args.output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "mini_rl_teacher.py"
    if args.replay:
        raw = args.replay.read_bytes().decode("utf-8")
    else:
        print("Running the unmodified default training script...", flush=True)
        completed = subprocess.run([sys.executable, "-u", str(source)], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode:
            sys.stderr.buffer.write(completed.stderr)
            raise SystemExit(completed.returncode)
        raw = completed.stdout.decode("utf-8")
    clean, intro, frames, final = parse_output(raw)
    (args.output / "terminal.txt").write_text(clean, encoding="utf-8")
    # Capture all printed frames, in order; do not select only successful iterations.
    scenes = [(intro, "01 / Reward model pretraining", 100, None)]
    scenes += [(lines, "02 / Joint training — all 101 printed frames", 8, i / (len(frames) - 1))
               for i, lines in enumerate(frames)]
    scenes += [(final, "03 / Actual evaluation — inspect measured distance and success rate", 175, None)]
    mp4 = args.output / "mini-rl-teacher.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
           "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt",
           "yuv420p", "-movflags", "+faststart", str(mp4)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as encoder:
        try:
            for lines, stage, repeat, progress in scenes:
                frame = render(lines, stage, font, progress)
                data = frame.tobytes()
                for _ in range(repeat):
                    encoder.stdin.write(data)
            encoder.stdin.close()
        except Exception:
            encoder.kill()
            raise
        if encoder.wait():
            raise RuntimeError("ffmpeg encoding failed")
    render(frames[-1], "Actual default run | Click to watch training and final evaluation", font, 1).save(
        args.output / "poster.png")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp4),
                    "-filter_complex", "fps=5,scale=864:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer",
                    "-loop", "0", str(args.output / "preview.gif")], check=True)
    metadata = {
        "source": "mini_rl_teacher.py",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "terminal_sha256": hashlib.sha256(clean.encode("utf-8")).hexdigest(),
        "training_command": "python3.11 -u mini_rl_teacher.py" if args.replay else f"{Path(sys.executable).name} -u mini_rl_teacher.py",
        "python": platform.python_version(), "pillow": pillow_version,
        "numpy": __import__("numpy").__version__,
        "training_frames": len(frames),
        "video_seconds": sum(s[2] for s in scenes) / FPS,
        "timing": "Condensed stdout replay: pretraining 4s, each printed training frame 0.32s, evaluation 7s",
        "evaluation_stdout": final,
        "note": "Metrics are from the captured default run, not a convergence guarantee. Replay mode assumes the supplied log matches this source and environment.",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"mp4": str(mp4), "seconds": metadata["video_seconds"], "frames": len(frames)}))


if __name__ == "__main__":
    main()
