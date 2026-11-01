# 基于姿势估计的康复动作评估

用 MediaPipe Pose 从视频里提取人体关键点坐标，得到逐帧 CSV，再按关节活动度对上下肢动作做分析。做这个东西的初衷是把康复训练里的动作质量判断从"治疗师肉眼看"变成可量化的指标。

原始数据是自己拍的 6 类动作示范（举臂、伸膝、坐立各分正确与错误），人像已做过打码处理。

## 目录

```
src/
  extract_coordinates.py   视频 -> 关键点 CSV
  assess_upper_limb.py     上肢动作分析
  assess_lower_limb.py     下肢动作分析
data/samples/              四个样例 CSV
docs/notes.md              最初的工作笔记
```

## 环境

Python 3.9 ~ 3.12。mediapipe 目前没有 3.13 以上的 wheel，装不上就换解释器。

```
pip install -r requirements.txt
```

## 数据格式

每个 CSV 一行一帧，列名是 `<侧>_<关节>_X|_Y|_Z|_visibility`，第一列 `frame` 是帧序号。

X/Y/Z 默认是 MediaPipe 的**世界坐标**（米制，原点在两侧髋关节中点），算距离和角度只能用这个。加 `--pixel` 会改成归一化的图像坐标，那样 z 只是相对深度，角度会失真。

`--landmarks` 决定输出哪些点，和数据集对齐：

| 参数 | 关键点 | 列数 |
|---|---|---|
| `upper` | 肩、肘、腕、小指、食指、拇指 | 49 |
| `lower` | 髋、膝、踝、脚跟、脚尖 | 41 |
| `all` | 全部 33 点 | 133 |

检测失败的帧会留空，不填 0 也不插值。

## 用法

提取坐标，可以直接给目录：

```
python src/extract_coordinates.py archive-WLU/Blurred/Arm -o csv --landmarks upper
```

分析单个文件：

```
python src/assess_lower_limb.py -f data/samples/knee_extension_correct_01.csv -s LEFT
python src/assess_upper_limb.py -f data/samples/arm_raise_correct_01.csv -s RIGHT
```

对比正确与错误动作，加 `--plot` 出柱状图：

```
python src/assess_upper_limb.py \
  -c data/samples/arm_raise_correct_01.csv \
  -i data/samples/arm_raise_incorrect_01.csv -s RIGHT --plot
```

## 评分说明

只对能自动测量的项目打分，全部是整数 0/1/2 档（协调性按下限 4 或 2 分）。

| | 项目 | 上限 |
|---|---|---|
| 下肢 | 屈髋、屈膝、踝背屈、协调性 | 10 |
| 上肢 | 肩、肘、腕、手、协调性 | 10 |

真正的 FMA-LE 满分 34、FMA-UE 满分 66，剩下的条目（反射、分离运动、协同模式、精细抓握）都要临床观察，单目摄像头测不出来，所以这里**不报总分**，只报"自动可测项得分 / 满分"。之前那一版把健康人强行抬到 30 分以上、患者压到 20 分以下，分数其实来自标签而不是动作，已经删掉。

档位阈值写在 `LEVELS` 里，取自常见关节活动度参考值，没有用标注数据标定过，换场景要重新定。

## 已知限制

- 没有金标准评分，给不出准确率。目前只能验证"两个动作算出来的指标确实不同"，不能证明"分数等于治疗师判断"。
- 世界坐标的 y 轴朝向在不同版本里说法不一，程序里取与竖直线夹角的绝对值绕开这个问题。代价是屈髋和伸髋无法区分。
- 手部关节在画面里小、遮挡多，腕和手指的活动度误差偏大，参考价值有限。
- 动作类型和患侧仍需人工指定，还没做到自动判别上下肢与动作类别。
- 原始数据集里有重复文件，同名视频在不同目录各存一份，没有统一整理。

## 数据

完整视频和 CSV（约 830 MB）没有放进仓库，单独发布在 Release 里。`data/samples/` 只留了四个小文件，够跑通流程。

视频涉及真实受试者，虽然做过人脸打码，引用或再分发之前请确认已获得本人同意。

## 许可

代码以 MIT 协议发布，见 LICENSE。
