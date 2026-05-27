---
name: action-camera-auto-trim
description: Automatically trim action camera videos (GoPro, DJI, Insta360) using self-adaptive computer vision algorithms and lossless FFmpeg stream copy.
metadata:
  tags: action-camera, video, trim, dji, gopro, insta360, python, cv2, ffmpeg
---

## When to use

Use this skill when you need to batch-process action camera footage (such as from DJI Osmo Action, GoPro, or Insta360) to automatically and losslessly trim off the "waste" footage at the beginning and end of recording. This is especially useful for removing selfie stick extension/retraction shake, button press impact vibrations, pocket/cover darkness, or slow transitional setup movements before the main action begins.

---

## Skill Architecture & Logic

This skill includes a self-adaptive computer vision (CV) algorithm implemented in Python. It does not require any hardware metadata parsing (which can be unreliable across camera models), but instead analyzes the frame-to-frame pixel differences (motion intensity) and overall frame brightness.

```
 Video Input ──> 1. Mid-segment analysis ──> Measure active sports baseline (M_baseline)
                   │
                   └──> 2. Auto-thresholding ──> Set custom shake threshold (M_baseline + margin)
                                                   │
 3. Forward Spike Interruption Scan ───────────────┘──> Cut immediately before the first massive end shake wave (retraction)
 4. Start Setup-Tail Scan ─────────────────────────┘──> Cut immediately after the last start handling shake wave (extension)
```

---

## Setup & Dependencies

The skill script relies on `opencv-python` and `numpy`. To ensure the environment is fully ready, run:
```bash
pip install opencv-python numpy
```

---

## How to use

The script is located at `scripts/auto_trim.py` inside this skill. You can run it directly:

### 1. Process a single video
```bash
python .agents/skills/action-camera-auto-trim/scripts/auto_trim.py -i <path_to_video.mp4>
```

### 2. Batch process an entire directory of videos
```bash
python .agents/skills/action-camera-auto-trim/scripts/auto_trim.py -i <path_to_folder>
```

### 3. Highly Recommended: Tuned parameters for Selfie Sticks (自拍杆通用推荐参数)
To aggressively cut out selfie stick extension (片头拉杆) and retraction (片尾收杆) phases:
```bash
python .agents/skills/action-camera-auto-trim/scripts/auto_trim.py -i <path_to_input> -m 1.5
```

---

## Detailed Command Line Parameters

| Option | Full Name | Default | Description |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | **Required** | Absolute or relative path to a video file or a folder of videos |
| `-o` | `--output` | Automatic | Destination folder to save trimmed videos (defaults to creating `trimmed/` under input path) |
| `-d` | `--dry-run` | Off | **Dry run mode**. Performs the CV analysis and prints recommended cuts without writing any output |
| `-s` | `--search-duration` | `15.0` | Seconds to analyze at the start and end of the video. Minimizes CPU decoding |
| `-b` | `--brightness` | `18.0` | Gray value threshold (0-255) for dark pocket/lens cap detection |
| `-m` | `--motion` | `3.0` | **Adaptive motion margin**. The sensitivity above the baseline. **Lower values cut more aggressively** |
| `-w` | `--window` | `1.5` | Required duration of stability in seconds to establish positive action |
| `-l` | `--min-duration` | `5.0` | Minimum allowed duration of the output video in seconds, to prevent over-cropping |

---

## Core Algorithm Implementation

The Python script `auto_trim.py` performs sequential reading using `cap.grab()` (which skips non-sampled frame decoding at a binary level) to optimize analysis speed by 1000x over regular seekers, finishing a 4K 100Mbps HEVC file analysis in a matter of seconds.
Lossless cut is achieved by calling the local `npx remotion ffmpeg` stream copy command (`-c copy`), keeping 100% original video bitstream quality without re-encoding.
