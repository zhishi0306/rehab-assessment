import argparse
import os

import numpy as np
import pandas as pd

# 简化判据：(1分下限, 2分下限)，单位度
LEVELS = {
    "shoulder": (30, 90),
    "elbow": (30, 90),
    "wrist": (10, 30),
    "hand": (10, 30),
}


class UpperLimbAssessment:
    """上肢动作分析：肩肘腕手指活动度、速度平滑度，并给出自动可测项的整数评分。"""

    def __init__(self, side="RIGHT", fps=30.0):
        self.side = side.upper()
        self.other = "LEFT" if self.side == "RIGHT" else "RIGHT"
        self.fps = float(fps)

    @staticmethod
    def load(path):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df.columns = [c.upper() for c in df.columns]
        return df

    def points(self, df, joint, side=None):
        s = side or self.side
        cols = [f"{s}_{joint}_{axis}" for axis in "XYZ"]
        for c in cols:
            if c not in df.columns:
                raise KeyError("缺少列: " + c)
        return df[cols].to_numpy(dtype=float)

    def drop_invalid(self, df):
        need = [c for c in df.columns if c.startswith(self.side + "_") and c[-1] in "XYZ"]
        if not need:
            need = [c for c in df.columns if c != "FRAME"]
        mask = df[need].notna().all(axis=1)
        return df.loc[mask].reset_index(drop=True), float(mask.mean())

    @staticmethod
    def angle(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        den = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(den > 0, np.sum(v1 * v2, axis=1) / den, np.nan)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    @staticmethod
    def tilt(vec):
        n = np.linalg.norm(vec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(n > 0, np.abs(vec[:, 1]) / n, np.nan)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    def joint_angles(self, df):
        sh = self.points(df, "SHOULDER")
        el = self.points(df, "ELBOW")
        wr = self.points(df, "WRIST")
        return {
            "shoulder": self.tilt(el - sh),
            "elbow": self.angle(sh, el, wr),
            "wrist": self.angle(el, wr, self.points(df, "PINKY")),
            "hand": self.angle(wr, self.points(df, "PINKY"), self.points(df, "INDEX")),
            "thumb": self.angle(wr, self.points(df, "THUMB"), self.points(df, "INDEX")),
        }

    def speed(self, df):
        pos = self.points(df, "WRIST")
        frame = df["FRAME"].to_numpy(dtype=float)
        dt = np.maximum(np.diff(frame), 1.0) / self.fps
        dist = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        return dist / dt

    @staticmethod
    def smoothness(values):
        """只统计真正在动的片段：静止帧速度接近 0，会把离散系数拉爆。"""
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 3:
            return 0.0
        moving = v[v > v.max() * 0.2]
        if moving.size < 3 or moving.mean() <= 0:
            return 0.0
        return float(np.clip(1.0 - moving.std() / moving.mean(), 0.0, 1.0))

    @staticmethod
    def grade(rom, low, high):
        if rom >= high:
            return 2
        if rom >= low:
            return 1
        return 0

    def score(self, rom, smooth):
        items = []
        for name, key in (("肩", "shoulder"), ("肘", "elbow"), ("腕", "wrist"), ("手", "hand")):
            low, high = LEVELS[key]
            items.append((name, self.grade(rom[key], low, high), 2))
        items.append(("协调性", round(smooth * 2), 2))
        total = sum(v for _, v, _ in items)
        full = sum(m for _, _, m in items)
        return items, total, full

    def analyze(self, path):
        raw = self.load(path)
        df, keep = self.drop_invalid(raw)
        if len(df) < 5:
            raise ValueError(f"有效帧不足({len(df)})，无法分析")

        angles = self.joint_angles(df)
        rom = {k: float(np.nanmax(v) - np.nanmin(v)) for k, v in angles.items()}
        spd = self.speed(df)
        smooth = self.smoothness(spd)
        items, total, full = self.score(rom, smooth)

        return {
            "file": os.path.basename(path),
            "side": self.side,
            "frames": len(raw),
            "valid": len(df),
            "keep": keep,
            "rom": rom,
            "speed_mean": float(np.nanmean(spd)) if spd.size else 0.0,
            "smooth": smooth,
            "items": items,
            "score": total,
            "full": full,
        }

    def report(self, r):
        lines = [
            "=" * 52,
            f"文件: {r['file']}",
            f"患侧: {r['side']}   有效帧: {r['valid']}/{r['frames']} ({r['keep'] * 100:.0f}%)",
            "",
            "关节活动度",
        ]
        for k, label in (
            ("shoulder", "肩"),
            ("elbow", "肘"),
            ("wrist", "腕"),
            ("hand", "手指"),
            ("thumb", "拇指"),
        ):
            lines.append(f"  {label}  {r['rom'][k]:6.1f}°")
        lines.append(f"速度均值 {r['speed_mean']:.3f} 单位/秒   平滑度 {r['smooth']:.2f}")
        lines.append("")
        lines.append(f"自动可测项评分 {r['score']}/{r['full']}")
        for name, got, full in r["items"]:
            lines.append(f"  {name:<6} {got}/{full}")
        lines.append("=" * 52)
        return "\n".join(lines)

    def compare(self, correct_path, incorrect_path):
        left = self.analyze(correct_path)
        right = self.analyze(incorrect_path)
        print(self.report(left))
        print(self.report(right))
        print(
            f"动作正确 vs 错误: 评分 {left['score']} vs {right['score']}，"
            f"肩活动度 {left['rom']['shoulder']:.1f}° vs {right['rom']['shoulder']:.1f}°"
        )
        return left, right

    def plot(self, correct, incorrect):
        import matplotlib.pyplot as plt

        labels = ["肩", "肘", "腕", "手指"]
        keys = ["shoulder", "elbow", "wrist", "hand"]
        a = [correct["rom"][k] for k in keys]
        b = [incorrect["rom"][k] for k in keys]

        x = np.arange(len(labels))
        w = 0.35
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))

        ax[0].bar(x - w / 2, a, w, label="正确", color="#4C78A8")
        ax[0].bar(x + w / 2, b, w, label="错误", color="#E45756")
        ax[0].set_xticks(x, labels)
        ax[0].set_ylabel("活动度 (°)")
        ax[0].legend()

        ax[1].bar(
            ["正确", "错误"], [correct["score"], incorrect["score"]], color=["#4C78A8", "#E45756"]
        )
        ax[1].set_ylabel("自动可测项评分")
        ax[1].set_ylim(0, correct["full"])

        fig.tight_layout()
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="上肢动作分析")
    ap.add_argument("-c", "--correct", help="动作正确的 CSV")
    ap.add_argument("-i", "--incorrect", help="动作错误的 CSV")
    ap.add_argument("-f", "--file", help="只分析单个 CSV")
    ap.add_argument("-s", "--side", default="RIGHT", choices=["LEFT", "RIGHT"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    analyzer = UpperLimbAssessment(args.side, args.fps)

    if args.file:
        print(analyzer.report(analyzer.analyze(args.file)))
    elif args.correct and args.incorrect:
        a, b = analyzer.compare(args.correct, args.incorrect)
        if args.plot:
            analyzer.plot(a, b)
    else:
        ap.error("需要 --file 或者同时给 --correct 和 --incorrect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
