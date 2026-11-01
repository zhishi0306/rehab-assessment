import argparse
import csv
import os

import cv2
import mediapipe as mp

POSE = mp.solutions.pose

# 与数据集 CSV 的列顺序对齐，left/right 交替
UPPER = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
         "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY",
         "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB"]
LOWER = ["LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE",
         "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
         "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"]

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")


def landmark_names(group):
    if group == "upper":
        return UPPER
    if group == "lower":
        return LOWER
    return [lm.name for lm in POSE.PoseLandmark]


def header_for(names):
    cols = ["frame"]
    for n in names:
        cols += [n + "_X", n + "_Y", n + "_Z", n + "_visibility"]
    return cols


def extract(video_path, out_path, names, world=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("打不开视频: " + video_path)

    index = {lm.name: int(lm) for lm in POSE.PoseLandmark}
    rows = []
    frame_no = 0
    hit = 0

    with POSE.Pose(static_image_mode=False, model_complexity=2,
                   smooth_landmarks=True, min_detection_confidence=0.5,
                   min_tracking_confidence=0.5) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            row = [frame_no]
            if res.pose_landmarks:
                hit += 1
                norm = res.pose_landmarks.landmark
                # 世界坐标是米制、以两髋中点为原点，算距离和角度只能用它
                if world and res.pose_world_landmarks:
                    body = res.pose_world_landmarks.landmark
                else:
                    body = norm
                for name in names:
                    i = index[name]
                    p = body[i]
                    row += [round(p.x, 6), round(p.y, 6), round(p.z, 6),
                            round(norm[i].visibility, 6)]
            else:
                row += [""] * (len(names) * 4)
            rows.append(row)
            frame_no += 1

    cap.release()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header_for(names))
        w.writerows(rows)

    return frame_no, hit


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, names in os.walk(p):
                for n in names:
                    if n.lower().endswith(VIDEO_EXT):
                        files.append(os.path.join(root, n))
        elif p.lower().endswith(VIDEO_EXT):
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description="提取人体关键点坐标到 CSV")
    ap.add_argument("input", nargs="+", help="视频文件或目录")
    ap.add_argument("-o", "--output", default="csv", help="输出目录")
    ap.add_argument("--landmarks", default="all",
                    choices=["all", "upper", "lower"],
                    help="all=33点(133列) upper=上肢12点(49列) lower=下肢10点(41列)")
    ap.add_argument("--pixel", action="store_true",
                    help="输出归一化图像坐标而不是米制世界坐标")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    names = landmark_names(args.landmarks)
    files = collect(args.input)
    if not files:
        print("没有找到视频")
        return 1

    total = len(files)
    for k, video in enumerate(files, 1):
        stem = os.path.splitext(os.path.basename(video))[0] + "_3d_coordinates.csv"
        out = os.path.join(args.output, stem)
        if args.skip_existing and os.path.exists(out):
            print("[%d/%d] 跳过 %s" % (k, total, stem))
            continue
        n, hit = extract(video, out, names, world=not args.pixel)
        print("[%d/%d] %s  帧 %d  有效 %d (%.0f%%)" %
              (k, total, stem, n, hit, 100.0 * hit / max(n, 1)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
