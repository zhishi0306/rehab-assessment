# 数据说明

`data/samples/` 里是四个样例，从完整数据集里挑出来的，够跑通流程：

| 文件 | 来源 | 帧数 |
|---|---|---|
| arm_raise_correct_01.csv | 举臂 / 正确 | 101 |
| arm_raise_incorrect_01.csv | 举臂 / 错误 | 101 |
| knee_extension_correct_01.csv | 伸膝 / 正确 | 87 |
| knee_extension_incorrect_01.csv | 伸膝 / 错误 | 64 |

## 完整数据

完整的视频（339 段，约 830 MB）和对应 CSV 没有放进仓库，单独放在 Release 里：<在此填 Release 链接>。

原始目录 `archive-WLU/Blurred/` 下按 `动作 / 正确或错误` 分文件夹，视频文件名与同名 CSV 的帧序一一对应。

## 几个要注意的地方

- 人像做过打码，但仍是真实受试者的训练画面，再分发前请确认授权。
- 原始数据里存在重复文件：同名同大小的视频在不同目录各存一份，`WLUcsv`、`WLU_finalcsv`、`archive-WLU` 三处的 CSV 也有大量重复，使用时按文件名去重即可。
- 早期版本导出的 CSV 是归一化图像坐标，不是米制。用 `src/extract_coordinates.py` 重新导出会得到世界坐标，两者不要混用。
