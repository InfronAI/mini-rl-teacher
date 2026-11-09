# 视频制作说明

[English](README.md) | **简体中文**

两版 Demo 均来自未修改的 `mini_rl_teacher.py` 默认 1,000 轮训练、每条轨迹 60 步、每轮 32 条轨迹的真实终端输出。它们是输出回放，不是桌面录屏或模拟训练动画。中英文版本使用同一次运行的数值、点阵位置和播放时序。

| 内容 | English | 简体中文 |
| --- | --- | --- |
| MP4 视频 | [Watch / download](mini-rl-teacher.en.mp4) | [观看 / 下载](mini-rl-teacher.zh-CN.mp4) |
| GIF 动图 | [Preview](preview.en.gif) | [预览](preview.zh-CN.gif) |
| 封面 | [Poster](poster.en.png) | [封面](poster.zh-CN.png) |
| 文字记录 | [英文翻译](terminal.en.txt) | [原始输出](terminal.txt) |
| 渲染记录 | [English](media.en.json) | [中文](media.zh-CN.json) |

两版 MP4 均为 1440 × 1000、H.264、25 fps，时长 43.32 秒。英文版翻译终端文字与画面说明；中文版保留原始终端输出，阶段标题和页脚也使用中文。训练脚本本身未修改，直接运行时仍输出中文。

全部 101 个训练画面按顺序播放，保留了最终 50/50 条评估轨迹通过验收的结果。预训练画面停留 4 秒，每个训练画面停留 0.32 秒，最终评估停留 7 秒。这些播放时长不代表实际执行耗时。脚本固定打印的成功提示也被保留，英文版予以翻译；评估标题提醒观众以实际指标判断成功与否。

录制使用 Python 3.11 和 NumPy 2.4.6。最终 50 条确定性策略轨迹的平均终点为 `(2.76, 1.81)`、平均终点距离 `0.31`、平均真实回报 `-35.60`，半径 `0.5` 内的成功率为 `100%`。全部 50 条评估轨迹通过终点验收；评估没有检验速度归零或后续持续停留。

[run.json](run.json) 保留原始录制的源码与输出 SHA-256、环境版本和评估结果。两种语言的渲染记录引用相同的原始输出哈希，并分别记录媒体文件哈希。英文文字记录是展示层翻译，不是另一次训练的输出。

## 复现

在仓库根目录、已激活的虚拟环境中执行：

```bash
python -m pip install numpy pillow
python scripts/record_demo.py --language both
```

另外安装 `ffmpeg` 并确保它位于 `PATH`。脚本会从 macOS、Linux 和 Windows 常用路径查找支持中文的字体；必要时使用 `--font /path/to/font.ttf` 指定，也支持 `.otf` 和 `.ttc`。字体只在本地使用，不随仓库分发。

录制脚本运行一次全新训练，捕获输出，再用 Pillow 和 ffmpeg 制作两版媒体。训练脚本会在预训练后将 `reward_model.npz` 保存到仓库根目录，最终评估后覆盖为最终奖励权重。录制脚本不传入 `--load`，已有奖励模型文件也不会跳过预训练。`--language` 支持 `en`、`zh-CN` 和 `both`，默认生成两版。生成文件会替换 `docs/demo/` 下的对应文件；录制新代码版本后，应同步更新两份 README 中的实测结果和源码哈希。

只从已有记录重新生成两版视频，无需训练：

```bash
python scripts/record_demo.py --replay docs/demo/terminal.txt --language both --output /tmp/mini-rl-teacher-replay
```

回放输入应使用原始中文记录，不应使用 `terminal.en.txt`。若同目录存在 `run.json`，脚本会验证输出哈希并保留原始录制信息；否则，无法确认的录制信息留空。当前渲染环境分别记录在 `media.en.json` 和 `media.zh-CN.json` 中。解析器要求默认 1,000 轮训练和当前输出格式；若英文画面中出现未翻译的中文，会报错而非混用两种语言。
