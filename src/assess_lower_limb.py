import argparse
import os

import numpy as np
import pandas as pd

# 简化判据：(1分下限, 2分下限)，单位度。阈值取自常见关节活动度参考值，
# 仅供自动筛查，不能替代临床量表评分。
LEVELS = {
    "hip": (20, 60),
    "knee": (15, 50),
    "ankle": (5, 15),
}


class LowerLimbAssessment:
    """下肢动作分析：关节活动度、对称性、速度平滑度，并给出自动可测项的整数评分。"""

    def __init__(self, side="LEFT", fps=30.0):
        self.side = side.upper()
        self.other = "RIGHT" if self.side == "LEFT" else "LEFT"
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
        cols = ["%s_%s_%s" % (s, joint, axis) for axis in "XYZ"]
        for c in cols:
            if c not in df.columns:
                raise KeyError("%s 里没有 %s" % (os.path.basename(c), c))
        return df[cols].to_numpy(dtype=float)

    def drop_invalid(self, df):
        """检测失败的帧坐标为空，直接丢，不做插值填充。"""
        need = [c for c in df.columns
                if c.startswith(self.side + "_") and c[-1] in "XYZ"]
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
        """向量与竖直线的夹角，取绝对值以避开坐标轴朝向的不确定性。"""
        n = np.linalg.norm(vec, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(n > 0, np.abs(vec[:, 1]) / n, np.nan)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))

    def hip_angle(self, df):
        # 有躯干点就用肩髋膝三点，没有就退回大腿与竖直线
        if self.side + "_SHOULDER_X" in df.columns:
            return self.angle(self.points(df, "SHOULDER"),
                              self.points(df, "HIP"),
                              self.points(df, "KNEE"))
        return self.tilt(self.points(df, "KNEE") - self.points(df, "HIP"))

    def joint_angles(self, df):
        hip = self.hip_angle(df)
        knee = self.angle(self.points(df, "HIP"),
                          self.points(df, "KNEE"),
                          self.points(df, "ANKLE"))
        ankle = self.angle(self.points(df, "KNEE"),
                           self.points(df, "ANKLE"),
                           self.points(df, "FOOT_INDEX"))
        return {"hip": hip, "knee": knee, "ankle": ankle}

    def speed(self, df):
        pos = self.points(df, "ANKLE")
        frame = df["FRAME"].to_numpy(dtype=float)
        # 丢帧之后相邻两行不再是相邻帧，按实际帧差换算
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
    def symmetry(df, side_a, side_b):
        span = []
        for s in (side_a, side_b):
            col = "%s_ANKLE_X" % s
            if col not in df.columns:
                return None
            span.append(float(df[col].max() - df[col].min()))
        hi = max(span)
        return float(min(span) / hi) if hi > 0 else None

    @staticmethod
    def grade(rom, low, high):
        if rom >= high:
            return 2
        if rom >= low:
            return 1
        return 0

    def score(self, rom, smooth, sym):
        items = []
        for name, key in (("屈髋", "hip"), ("屈膝", "knee"), ("踝背屈", "ankle")):
            low, high = LEVELS[key]
            items.append((name, self.grade(rom[key], low, high), 2))
        items.append(("协调性", int(round(smooth * 4)), 4))
        total = sum(v for _, v, _ in items)
        full = sum(m for _, _, m in items)
        return items, total, full

    def analyze(self, path):
        raw = self.load(path)
        df, keep = self.drop_invalid(raw)
        if len(df) < 5:
            raise ValueError("有效帧不足(%d)，无法分析" % len(df))

        angles = self.joint_angles(df)
        rom = {k: float(np.nanmax(v) - np.nanmin(v)) for k, v in angles.items()}
        spd = self.speed(df)
        smooth = self.smoothness(spd)
        sym = self.symmetry(df, self.side, self.other)
        items, total, full = self.score(rom, smooth, sym)

        return {
            "file": os.path.basename(path),
            "side": self.side,
            "frames": len(raw),
            "valid": len(df),
            "keep": keep,
            "rom": rom,
            "speed_mean": float(np.nanmean(spd)) if spd.size else 0.0,
            "smooth": smooth,
            "symmetry": sym,
            "items": items,
            "score": total,
            "full": full,
        }

    def report(self, r):
        lines = ["=" * 52,
                 "文件: %s" % r["file"],
                 "患侧: %s   有效帧: %d/%d (%.0f%%)" %
                 (r["side"], r["valid"], r["frames"], r["keep"] * 100),
                 "",
                 "关节活动度"]
        for k, label in (("hip", "髋"), ("knee", "膝"), ("ankle", "踝")):
            lines.append("  %s  %6.1f°" % (label, r["rom"][k]))
        lines.append("速度均值 %.3f 单位/秒   平滑度 %.2f" %
                     (r["speed_mean"], r["smooth"]))
        if r["symmetry"] is not None:
            lines.append("左右对称性 %.2f" % r["symmetry"])
        lines.append("")
        lines.append("自动可测项评分 %d/%d" % (r["score"], r["full"]))
        for name, got, full in r["items"]:
            lines.append("  %-6s %d/%d" % (name, got, full))
        lines.append("=" * 52)
        return "\n".join(lines)

    def compare(self, correct_path, incorrect_path):
        left = self.analyze(correct_path)
        right = self.analyze(incorrect_path)
        print(self.report(left))
        print(self.report(right))
        print("动作正确 vs 错误: 评分 %d vs %d，髋活动度 %.1f° vs %.1f°" %
              (left["score"], right["score"],
               left["rom"]["hip"], right["rom"]["hip"]))
        return left, right

    def plot(self, correct, incorrect):
        import matplotlib.pyplot as plt

        labels = ["屈髋", "屈膝", "踝背屈"]
        keys = ["hip", "knee", "ankle"]
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

        ax[1].bar(["正确", "错误"], [correct["score"], incorrect["score"]],
                  color=["#4C78A8", "#E45756"])
        ax[1].set_ylabel("自动可测项评分")
        ax[1].set_ylim(0, correct["full"])

        fig.tight_layout()
        plt.show()


def main():
    ap = argparse.ArgumentParser(description="下肢动作分析")
    ap.add_argument("-c", "--correct", help="动作正确的 CSV")
    ap.add_argument("-i", "--incorrect", help="动作错误的 CSV")
    ap.add_argument("-f", "--file", help="只分析单个 CSV")
    ap.add_argument("-s", "--side", default="LEFT", choices=["LEFT", "RIGHT"])
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    analyzer = LowerLimbAssessment(args.side, args.fps)

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
