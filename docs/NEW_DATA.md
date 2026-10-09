# 使用自己的采集数据

新配置把输入文件、声速、采样频率和归一化目标放在 JSON 中。每次重建先准备本次数据副本，自动推导探头数和时间长度，并为这一帧重新计算幅值增益。附带的 `data/pose000` 和示例结果保持不变。

## 一条命令开始

如果信号是 `[N,T]`、坐标是以米为单位的 `[N,3]`，可以直接覆盖默认配置的采集字段：

```bash
./run.sh reconstruct.py --signal /path/to/processed_rf.npy \
  --sensors /path/to/sensor_xyz.npy --sound-speed 1480 \
  --sampling-frequency 20000000 --samples auto --output outputs/new_frame
```

`--sound-speed` 单位为 m/s，`--sampling-frequency` 为 Hz。`--samples auto` 自动读取长度，整数值只做一致性检查。常用覆盖还包括 `--coordinate-unit mm`、`--signal-layout time_sensor` 和 `--target-rms 15.941105800921543`；其余导入选项在 JSON 中设置。

需要保存完整配置时，使用以下方式：

复制完整配置，然后编辑新文件中的 `input`、`acquisition` 和可选的 `normalization`、`sensor_split`：

```bash
cp configs/pose000.json configs/new_frame.json
```

以下是需要替换的字段示例；保留复制文件中的其余训练、细分、正则和导出设置：

```json
{
  "input": {
    "signal": "/path/to/processed_rf.npy",
    "sensors": "/path/to/sensor_xyz.npy",
    "signal_key": null,
    "sensors_key": null,
    "signal_layout": "sensor_time",
    "sensor_layout": "sensor_xyz",
    "coordinate_unit": "m"
  },
  "acquisition": {
    "sound_speed_m_s": 1485.0,
    "sampling_frequency_hz": 20000000.0,
    "samples": null,
    "t0_seconds": 0.0
  },
  "normalization": {
    "target_rms": 15.941105800921543
  },
  "sensor_split": {
    "file": null,
    "validation_fraction": 0.125,
    "seed": 20261008
  }
}
```

然后运行：

```bash
./run.sh reconstruct.py --config configs/new_frame.json --output outputs/new_frame
```

输入准备是流程的第一步，生成 `outputs/new_frame/data/`；后续训练、响应拟合、导出和预览使用该副本中的数据与采样信息。`--dry-run` 可先查看计划命令。换输入或采集参数时使用新的输出目录，不要对旧运行强行续跑。

可以先运行同一命令并加 `--prepare-only`，检查数据后再以相同参数加 `--resume`、去掉 `--prepare-only` 继续训练：

```bash
./run.sh reconstruct.py --config configs/new_frame.json --output outputs/new_frame --prepare-only
python scripts/check_environment.py --data-dir outputs/new_frame/data
./run.sh reconstruct.py --config configs/new_frame.json --output outputs/new_frame --resume
```

相对路径均以项目根目录为基准，绝对路径可以指向项目外的只读输入。通过完整配置启动时，不需要手动设置 `data_dir` 或 `SLINGBAG_DATA_DIR`。

## 输入文件和轴序

信号必须是已经完成所需采集预处理的**有符号 RF**。本项目不会替你重建 ADC 处理链，也不会默认滤波、取绝对值、翻转符号、减去均值或调整时间零点。

| 字段 | 可选值与含义 |
|---|---|
| `input.signal` | RF 文件路径，包含有限实数二维数组 |
| `input.sensors` | 探头坐标文件路径，包含有限实数二维数组 |
| `signal_layout: "sensor_time"` | 信号原文件为 `[N,T]` |
| `signal_layout: "time_sensor"` | 原文件为 `[T,N]`，准备时转置为 `[N,T]` |
| `sensor_layout: "sensor_xyz"` | 探头坐标原文件为 `[N,3]` |
| `sensor_layout: "xyz_sensor"` | 原文件为 `[3,N]`，准备时转置为 `[N,3]` |
| `coordinate_unit` | `"m"`、`"mm"` 或 `"cm"`；准备后统一为米 |
| `signal_key` / `sensors_key` | NPZ 或 MAT 中的数组名称；NPY 和纯文本通常填 `null` |
| `signal_skiprows` / `sensors_skiprows` | 可选非负整数，TXT/CSV 开头需要跳过的行数，默认 `0` |
| `signal_delimiter` / `sensors_delimiter` | 可选单字符分隔符；默认 TXT 按空白分隔，CSV 按逗号分隔 |

支持 NPY、NPZ、TXT、CSV 和 SciPy 可读取的 MAT。默认文本文件应只含数值；有开头标题时用 `*_skiprows` 明确跳过，剩余数据不能混有通道名称或单位字符串。NPZ/MAT 中有多个变量时应明确指定键；只有一个变量时可以使用 `null`。MAT v7.3/HDF5 不受支持，先在已有工具中转换为 NPY 或非 v7.3 MAT。

第 `i` 行 RF 必须对应第 `i` 行探头坐标。布局选项只转置数组，单位选项只缩放坐标，不会重新排列通道，也不会进行刚体配准。坐标应已位于需要重建的 XYZ 坐标系；形状正确不能证明通道配对正确。

## 声学采样参数

`sound_speed_m_s` 和 `sampling_frequency_hz` 必须为正数，分别使用 m/s 和 Hz。采样间隔由 `dt_seconds = 1 / sampling_frequency_hz` 得到，并传给声学前向及导出。

`samples: null` 从 RF 时间轴自动推导 `T`；填写正整数则检查输入长度与它一致。这个字段**不会裁剪、补零或重采样**。探头数 `N` 从 RF 和坐标共同推导，两者必须匹配。

当前只支持 `t0_seconds: 0`，时间采样为 `t[n] = n / sampling_frequency_hz`。如果输入记录相对激发有非零时间偏移，需要先在上游明确校正；不能通过填写一个非零数值来隐式改变波形解释。

本次参数化不改变重建视野：XYZ 均为 `[-20,20] mm`，导出 `400×400×400` 体素，间距 `0.1 mm`，首个体素中心 XYZ 为 `[-19.95,-19.95,-19.95] mm`，数组轴序为 **ZYX**。视野和体素网格暂不能通过输入配置修改。

共享 FIR 的长度和滞后用采样点计数；改变采样频率也会改变同样 tap 数对应的物理时间宽度。原配置在 pose000 上得到的正则强度、σ 和阶段步数是起点，不保证新帧保有同样质量。

## 自动 RMS 增益

准备时按整帧所有 `N×T` 个样本计算：

```text
measured_rms = sqrt(mean(signal_original ** 2))
global_gain = normalization.target_rms / measured_rms
signal_normalized = signal_original * global_gain
```

计算使用输入信号的平方均值，而非标准差；不先减均值。全零、非有限输入或非正目标 RMS 会被拒绝。每帧所有探头共用一个增益，因此保留探头间相对幅值。原文件只读，准备后的训练数组为 float32。

默认目标 `15.941105800921543` 沿用附带实验的数值幅值尺度，**不是绝对声压标定**。新帧必须重新计算 `global_gain`，不能复制 pose000 的 `0.2612356995077034`。训练直接读取准备后的 `signal_normalized.npy`，不要再次缩放；恢复原输入幅值的 RF 预测时除以这一帧的 `global_gain`。

这个全帧 RMS 使用了所有探头，不能把它描述为只用训练子集的预处理。保留探头参与参数选择后属于调参数据；全探头重拟与最终响应校准后，两组探头都属于拟合数据。

## 探头划分

`sensor_split.file: null` 根据 `validation_fraction` 和 `seed` 创建划分：先计算 `round(N × validation_fraction)`，再限制到 `[1,N−1]`；使用随机排列的前这么多个索引作为保留组，再将两组分别排序。至少需要两个探头，比例必须位于 `(0,1)`。若已有划分，将 `file` 指向 NPZ，包含一维整数数组 `train` 与 `validation`。

索引按转置后的 `[N,T]` 行编号，从 `0` 开始。两组必须各自无重复、互不重叠，并恰好覆盖 `0…N-1`。通道顺序或探头数量变化后，不能继续套用原 pose000 的 896/128 索引。

## 先准备和检查，不开始训练

```bash
python scripts/prepare_data.py --config configs/new_frame.json --output outputs/prepared
python scripts/check_environment.py --data-dir outputs/prepared
```

增加 `--cuda` 会检查工具链与设备，并运行一个短算术测试，不开始训练。`--data` 是 `--data-dir` 的兼容别名。

准备目录应包含以下文件：

| 文件 | 内容 |
|---|---|
| `signal_original.npy` | 转为 `[N,T]` 的有符号输入 RF，未应用 RMS 增益 |
| `signal_normalized.npy` | 同形状 float32 RF，等于 original × global_gain |
| `sensor_locations.npy` | `[N,3]` XYZ，单位米 |
| `metadata.json` | 增益、目标 RMS、声速、采样频率、采样间隔、长度和固定物理网格 |
| `split.npz` | `train` / `validation` 探头索引 |

检查器逐项验证有限性、数组维度、实际 RMS、增益关系、`sampling_frequency_hz × dt_seconds = 1`、时间长度、网格和探头划分。较旧的 pose000 metadata 没有显式频率或新归一化字段时，仍可由 `dt_seconds` 及历史 RMS 信息验证。

重建完成后可预览本次体积：

```bash
python scripts/preview.py --volume outputs/new_frame/final/volume_zyx.npy \
  --metadata outputs/new_frame/data/metadata.json --output outputs/new_frame/preview.png
```

三个 MIP 共用完整体积最大值，线性显示，无百分位截顶或显示平滑。检查器和预览使用 CPU，不依赖旧实验目录或参考重建体积。
