#!/usr/bin/env python3
import os
import sys
import time
import argparse
import subprocess
import glob
import shutil
import numpy as np
import cv2

# Force UTF-8 encoding on standard output/error to prevent UnicodeEncodeError with emojis or Chinese characters on Windows
try:
    if sys.platform.startswith("win"):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Default configuration parameters
DEFAULT_SEARCH_DURATION = 15.0  # Seconds to search at start/end
DEFAULT_BRIGHTNESS_THRESH = 18.0  # Pocket/cap detection (0-255)
DEFAULT_MOTION_MARGIN = 3.0  # Adaptive margin over mid-segment baseline
DEFAULT_WINDOW_SIZE = 1.5  # Required duration of stability in seconds
DEFAULT_MIN_DURATION = 5.0  # Minimum output duration
SAMPLE_FPS = 10  # Sample rate (fps) to speed up frame decoding

def parse_args():
    parser = argparse.ArgumentParser(
        description="DJI Action 4 视频 AI/算法全自动批量去片头片尾脱水工具"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="输入视频 file 路径，或者包含视频的文件夹路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出文件夹路径。如果不指定，默认在输入文件夹下创建 'trimmed' 文件夹"
    )
    parser.add_argument(
        "-s", "--search-duration", type=float, default=DEFAULT_SEARCH_DURATION,
        help=f"在片头和片尾搜索垃圾片段的最大时长（秒），默认 {DEFAULT_SEARCH_DURATION} 秒"
    )
    parser.add_argument(
        "-b", "--brightness", type=float, default=DEFAULT_BRIGHTNESS_THRESH,
        help=f"口袋/遮挡检测的亮度阈值（0-255），默认 {DEFAULT_BRIGHTNESS_THRESH}"
    )
    parser.add_argument(
        "-m", "--motion", type=float, default=DEFAULT_MOTION_MARGIN,
        help=f"自适应运动阈值裕量（正片基准之上允许的晃动裕量，越小剪切越激进），默认 {DEFAULT_MOTION_MARGIN}"
    )
    parser.add_argument(
        "-w", "--window", type=float, default=DEFAULT_WINDOW_SIZE,
        help=f"判断画面转为平稳所需的连续稳定时长（秒），默认 {DEFAULT_WINDOW_SIZE} 秒"
    )
    parser.add_argument(
        "-l", "--min-duration", type=float, default=DEFAULT_MIN_DURATION,
        help=f"脱水后正片的最小保留时长（秒），默认 {DEFAULT_MIN_DURATION} 秒"
    )
    parser.add_argument(
        "-d", "--dry-run", action="store_true",
        help="仅分析并打印裁剪时间点，不进行实际剪切"
    )
    return parser.parse_args()

def get_ffmpeg_path():
    """
    检查并返回 ffmpeg 执行命令。由于用户本地是 Remotion 环境，优先使用 npx remotion ffmpeg，
    如果不行则尝试系统自带的 ffmpeg。
    """
    # 尝试运行 npx remotion ffmpeg
    try:
        res = subprocess.run(
            ["npx", "remotion", "ffmpeg", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True, timeout=5
        )
        if res.returncode == 0:
            return ["npx", "remotion", "ffmpeg"]
    except Exception:
        pass
    
    # 尝试系统 ffmpeg
    try:
        res = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=True, timeout=5
        )
        if res.returncode == 0:
            return ["ffmpeg"]
    except Exception:
        pass
    
    print("[警告] 未在系统中找到 ffmpeg 或 npx remotion ffmpeg。视频剪切功能可能不可用！", file=sys.stderr)
    return None

def analyze_segment(video_path, start_time, duration, sample_fps=SAMPLE_FPS):
    """
    分析指定时间段内的视频帧，返回每一帧的 (时间戳, 亮度, 运动强度)。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_dur = total_frames / fps
    
    # 确保请求的时间段在视频范围内
    if start_time < 0:
        start_time = 0
    if start_time + duration > video_dur:
        duration = video_dur - start_time
        
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    # 设定帧读取间隔，以达到 sample_fps 的采样率
    frame_step = max(1, int(fps / sample_fps))
    
    frames_data = []
    prev_gray = None
    
    # 核心极速优化：仅 seek 一次定位到起始帧，随后顺序读取。
    # 针对跳过的帧调用 cap.grab() (仅解包不解码)，速度比频繁 set() 提升数百倍！
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame):
        # 仅当处于采样步长帧时，进行完整读取和 CV 处理；其余帧直接快速抓取跳过
        if (frame_idx - start_frame) % frame_step != 0:
            cap.grab()
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
            
        t = frame_idx / fps
        
        # 1. 极速缩放画面以过滤高频噪音并加速计算 (缩放到 160x90)
        small_frame = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 2. 计算平均亮度
        brightness = float(np.mean(gray))
        
        # 3. 计算帧间运动强度 (帧差法)
        motion = 0.0
        if prev_gray is not None:
            # 帧差平均绝对值
            motion = float(np.mean(np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16))))
            
        prev_gray = gray
        frames_data.append({
            'time': t,
            'brightness': brightness,
            'motion': motion
        })
        
    cap.release()
    return frames_data

def find_trim_points(video_path, search_dur, brightness_th, motion_th, window_sec, min_dur):
    """
    智能分析并计算视频的 T_start 和 T_end。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[错误] 无法打开视频文件: {video_path}")
        return None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_dur = total_frames / fps
    cap.release()
    
    print(f"视频总时长: {video_dur:.2f} 秒 (帧率: {fps:.2f})")
    
    if video_dur <= min_dur:
        print("[提示] 视频本身太短，无需裁剪。")
        return 0.0, video_dur
        
    # 如果视频长度不足以支持双侧 search_duration，则按比例缩减搜索范围
    actual_search_dur = search_dur
    if video_dur < (search_dur * 2 + min_dur):
        actual_search_dur = max(1.0, (video_dur - min_dur) / 2.0)
        print(f"[提示] 视频较短，调整单侧分析范围为 {actual_search_dur:.2f} 秒")
        
    # 全局定义稳定区间所需的采样帧数，避免局部作用域定义导致 NameError
    req_frames = int(window_sec * SAMPLE_FPS)
        
    # ---------------- 1. 分析片头 ----------------
    print(f"正在分析片头前 {actual_search_dur:.2f} 秒...")
    start_data = analyze_segment(video_path, 0, actual_search_dur)
    
    # 引入【自拍杆拉伸/起手快门抖动后置判定】：
    # 寻找片头搜索范围内最后一个“明显的起手机位调整/强晃动波峰”。
    # 正片应该在最后一个高频动作彻底结束，并且加上一段稳定期才算真正开始。
    last_spike_time = 0.0
    start_spike_found = False
    
    # 判定起手大波动的判定线：自适应运动阈值的 1.15 倍，且至少为 18.0
    start_spike_th = max(18.0, motion_th * 1.15)
    
    for frame in start_data:
        # 如果帧的运动强度显著大于稳定正片，说明自拍杆仍在拉伸、旋转或相机被手挪动
        if frame['motion'] > start_spike_th:
            last_spike_time = frame['time']
            start_spike_found = True
            
    t_start = 0.0
    if start_spike_found:
        # 正片起点 = 最后一个起手大波峰 + 1.5秒的阻尼稳定缓冲（以彻底剪掉拉杆到位后的微小余震）
        t_start = last_spike_time + 1.5
        print(f"[分析] 智能检测到片头最后一个动作/晃动波峰位于 {last_spike_time:.2f} 秒，正片起点推延至 {t_start:.2f} 秒（含 1.5s 稳定延迟）")
    else:
        # 如果压根没有任何晃动波峰（如开机就绝对平稳），则退回传统的稳定区间检测
        stable_found = False
        for i in range(len(start_data) - req_frames + 1):
            window = start_data[i : i + req_frames]
            good_frames = 0
            for frame in window:
                is_bright = frame['brightness'] > brightness_th
                is_stable = frame['motion'] < motion_th
                if is_bright and is_stable:
                    good_frames += 1
            if good_frames / req_frames >= 0.85:
                t_start = window[0]['time']
                stable_found = True
                break
                
        if not stable_found:
            print("[分析] 未在开头检测到明显稳定的分界点，默认保留开头。")
            t_start = 0.0
        else:
            print(f"[分析] 检测到片头稳定点: {t_start:.2f} 秒")
        
    # ---------------- 2. 分析片尾 ----------------
    print(f"正在分析片尾后 {actual_search_dur:.2f} 秒...")
    end_segment_start_time = video_dur - actual_search_dur
    end_data = analyze_segment(video_path, end_segment_start_time, actual_search_dur)
    
    t_end = video_dur
    stable_found = False
    
    # 引入【自拍杆收回/镜头拿取前置波峰判定】：
    # 从前向后扫描片尾段，寻找第一个“持续性剧烈晃动”的起点。
    # 如果在片尾段的中前部就发现了剧烈晃动，说明正片在这个时间点就已经结束了，
    # 后面发生的任何平静（如手握相机停下看屏幕）都应该被彻底剪掉。
    spike_found = False
    spike_time = None
    
    # 判定剧烈晃动的判定线：自适应运动阈值的 1.25 倍，且至少为 18.0
    shake_spike_th = max(18.0, motion_th * 1.25)
    
    # 寻找连续 4 帧（约 0.4 秒）均超标的晃动起点，以过滤单帧颠簸噪声
    for i in range(len(end_data) - 4):
        sub_seq = end_data[i : i + 4]
        is_spike = all(frame['motion'] > shake_spike_th for frame in sub_seq)
        if is_spike:
            spike_time = sub_seq[0]['time']
            spike_found = True
            break
            
    if spike_found:
        # 如果检测到了收尾动作，正片在此晃动发生前 1.0 秒截止（含安全缓冲以应对慢速回拉过渡期）
        t_end = max(t_start + min_dur, spike_time - 1.0)
        stable_found = True
        print(f"[分析] 智能检测到片尾收杆/拿取动作始于 {spike_time:.2f} 秒，正片提前截止于 {t_end:.2f} 秒（含 1.0s 安全缓冲）")
    else:
        # 如果没有突发的晃动波峰，则退回到传统的倒序稳定区间检索
        for i in range(len(end_data), req_frames - 1, -1):
            window = end_data[i - req_frames : i]
            good_frames = 0
            for frame in window:
                is_bright = frame['brightness'] > brightness_th
                is_stable = frame['motion'] < motion_th
                if is_bright and is_stable:
                    good_frames += 1
                    
            if good_frames / req_frames >= 0.85:
                t_end = window[-1]['time']
                stable_found = True
                break
                
        if not stable_found:
            print("[分析] 未在结尾检测到明显的收网动作，默认保留结尾。")
            t_end = video_dur
        else:
            print(f"[分析] 检测到片尾收拾点: {t_end:.2f} 秒 (切除最后 {(video_dur - t_end):.2f} 秒)")
        
    # ---------------- 3. 安全性检查 ----------------
    # 确保裁剪后的正片时长仍然合理
    if t_end - t_start < min_dur:
        print(f"[警告] 裁剪后视频过短 (仅 {t_end - t_start:.2f} 秒)，为避免错剪，将不进行裁剪。")
        return 0.0, video_dur
        
    return t_start, t_end

def trim_video(ffmpeg_cmd, input_path, output_path, t_start, t_end):
    """
    调用 FFmpeg 进行极速无损剪切。
    """
    if ffmpeg_cmd is None:
        print("[错误] 未找到 ffmpeg，无法执行剪切。请手动安装 ffmpeg 后重试。")
        return False
        
    # 构建 ffmpeg 命令行
    # -y: 覆盖输出文件
    # -ss before -i: 极速定位关键帧
    # -c copy: 拷贝流，不重新编码，实现 100% 画质 and 秒级剪切
    cmd = []
    # 如果是 npx remotion ffmpeg，需要展开
    cmd.extend(ffmpeg_cmd)
    
    # ffmpeg 时间参数必须精确格式化
    cmd.extend([
        "-y",
        "-ss", f"{t_start:.3f}",
        "-to", f"{t_end:.3f}",
        "-i", input_path,
        "-c", "copy",
        output_path
    ])
    
    print(f"执行裁剪命令: {' '.join(cmd)}")
    
    try:
        # 在 Windows 上，如果是 npx 开头，通常需要 shell=True
        use_shell = os.name == 'nt'
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=use_shell, check=True
        )
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return True
    except subprocess.CalledProcessError as e:
        print(f"[裁剪失败] FFmpeg 返回错误代号: {e.returncode}")
        print(f"错误输出:\n{e.stderr.decode('utf-8', errors='ignore')}")
    except Exception as e:
        print(f"[裁剪失败] 出现未知异常: {str(e)}")
        
    return False

def process_file(video_path, output_dir, args, ffmpeg_cmd):
    filename = os.path.basename(video_path)
    print(f"\n==========================================")
    print(f"[处理] 正在处理文件: {filename}")
    print(f"==========================================")
    
    cap = cv2.VideoCapture(video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_dur = total_frames / fps
    cap.release()
    
    # 自适应计算：分析正片中段（50% 进度处前后 2s）的运动强度基准
    print(f"正在分析视频中段运动基准以启用自适应阈值...")
    mid_start = max(0.0, orig_dur / 2.0 - 1.0)
    mid_dur = min(2.0, orig_dur)
    mid_data = analyze_segment(video_path, mid_start, mid_dur)
    
    motion_values = [f['motion'] for f in mid_data if f['motion'] > 0.0]
    if motion_values:
        # 使用第 70 百分位数作为稳定的正片基准，排除偶尔的突然晃动
        mid_baseline = float(np.percentile(motion_values, 70))
    else:
        mid_baseline = 15.0  # 默认兜底值
        
    # 自适应裁剪阈值 = 中段正片基准 + 裕量（通过命令行 -m 参数传递，默认 3.0）
    adapted_motion_th = max(5.0, mid_baseline + args.motion)
    print(f"【自适应抖动检测】中段正片基准: {mid_baseline:.2f} | 裕量: {args.motion:.2f} | 最终裁剪阈值: {adapted_motion_th:.2f}")
    
    start_time = time.time()
    t_start, t_end = find_trim_points(
        video_path,
        args.search_duration,
        args.brightness,
        adapted_motion_th,  # 传入自适应计算后的阈值！
        args.window,
        args.min_duration
    )
    
    if t_start is None or t_end is None:
        print(f"[-] 视频分析失败。")
        return False
        
    cut_start_len = t_start
    cut_end_len = orig_dur - t_end
    
    if cut_start_len <= 0.05 and cut_end_len <= 0.05:
        print("[提示] 该视频前后均无明显垃圾片段，无需进行裁剪剪切。")
        # 复制原文件到输出目录以保持批处理完整性
        out_path = os.path.join(output_dir, filename)
        if not args.dry_run:
            print(f"复制原文件到输出文件夹...")
            shutil.copy2(video_path, out_path)
            print(f"[+] 处理完成！已保持原样输出。")
        return True
        
    out_path = os.path.join(output_dir, filename)
    print(f"[决策] 智能识别决策:")
    print(f"   - 片头切除: {cut_start_len:.2f} 秒")
    print(f"   - 片尾切除: {cut_end_len:.2f} 秒")
    print(f"   - 正片区间: {t_start:.2f}s 至 {t_end:.2f}s (共计 {t_end - t_start:.2f} 秒)")
    
    if args.dry_run:
        print("[Dry Run] 仅进行算法分析，不写入新文件。")
        return True
        
    # 执行实际裁剪
    print("[剪切] 正在进行无损快速剪切...")
    success = trim_video(ffmpeg_cmd, video_path, out_path, t_start, t_end)
    elapsed = time.time() - start_time
    
    if success:
        orig_size = os.path.getsize(video_path) / (1024 * 1024)
        new_size = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[+] 裁剪成功！耗时: {elapsed:.2f} 秒")
        print(f"   - 原文件大小: {orig_size:.2f} MB")
        print(f"   - 脱水后大小: {new_size:.2f} MB")
        print(f"   - 输出路径: {out_path}")
        return True
    else:
        print("[-] 视频裁剪失败！")
        return False

def main():
    args = parse_args()
    
    # 查找本地 FFmpeg 工具
    ffmpeg_cmd = get_ffmpeg_path()
    if ffmpeg_cmd is None and not args.dry_run:
        print("[-] 无法找到可用的 FFmpeg！请先在系统中安装 ffmpeg，或者确保 node 项目里安装了 Remotion 包。")
        sys.exit(1)
        
    input_path = os.path.abspath(args.input)
    
    # 确定处理的文件列表
    video_extensions = ["*.mp4", "*.MOV", "*.mov", "*.MP4", "*.mkv"]
    files_to_process = []
    
    if os.path.isdir(input_path):
        # 扫描整个文件夹
        print(f"[扫描] 正在扫描文件夹: {input_path}")
        for ext in video_extensions:
            files_to_process.extend(glob.glob(os.path.join(input_path, ext)))
        # 去重
        files_to_process = sorted(list(set(files_to_process)))
        
        # 确定输出文件夹
        if args.output:
            output_dir = os.path.abspath(args.output)
        else:
            output_dir = os.path.join(input_path, "trimmed")
    else:
        # 处理单个文件
        if os.path.exists(input_path):
            files_to_process.append(input_path)
            
            # 确定输出文件夹
            if args.output:
                output_dir = os.path.abspath(args.output)
            else:
                output_dir = os.path.join(os.path.dirname(input_path), "trimmed")
        else:
            print(f"[-] 输入路径不存在: {input_path}")
            sys.exit(1)
            
    if not files_to_process:
        print("[-] 未找到任何符合格式的视频文件 (支持 .mp4, .mov, .mkv)")
        sys.exit(0)
        
    print(f"[*] 共找到 {len(files_to_process)} 个视频文件待处理。")
    
    # 创建输出目录
    if not args.dry_run:
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出文件夹已创建: {output_dir}")
        
    success_count = 0
    total_start_time = time.time()
    
    for video_file in files_to_process:
        if os.path.basename(video_file).startswith("._"):
            continue
        try:
            if process_file(video_file, output_dir, args, ffmpeg_cmd):
                success_count += 1
        except Exception as e:
            print(f"[-] 处理文件 {os.path.basename(video_file)} 时发生异常: {str(e)}")
            
    total_elapsed = time.time() - total_start_time
    print(f"\n==========================================")
    print(f"[完毕] 所有任务处理完毕！")
    print(f"==========================================")
    print(f"📁 视频输入路径: {input_path}")
    print(f"📁 视频输出路径: {output_dir}")
    print(f"[+] 成功处理: {success_count} / {len(files_to_process)} 个视频")
    print(f"[-] 总运行耗时: {total_elapsed:.2f} 秒")
    print(f"==========================================")

if __name__ == "__main__":
    main()
