# SlingBAG_3D-PanoPACT

**基于连续解析 Gaussian 点云的三维光声迭代重建。**

[English](README.en.md) · [原始 SlingBAG 仓库](https://github.com/JaegerCQ/SlingBAG) · [原始 SlingBAG 论文](https://www.nature.com/articles/s41467-025-66855-w) · [方法说明](docs/METHOD.md)

本项目从随机 Gaussian 点云开始，利用实测光声 RF 信号优化三维初始压力场。仓库包含重建代码、一帧 `group03_pose000` 实测信号和探头坐标、运行配置，以及已完成的点云和体积结果。复制本目录即可独立使用，无需访问此前的 PA-SfM-NC 或实验目录。

这是基于 SlingBAG 思路的算法变体：采用连续解析 Gaussian 前向、物理场 TV＋曲率正则、守恒细分和分阶段尺度调整。球心 XYZ 在开放更新后连续可微优化，σ 在各阶段内固定。新数据的输入路径、声速、采样频率和时间点数统一由配置控制，每次新运行自动计算这一帧的全局归一化增益。

## Display of results / 结果展示

![随机初始化重建的三个方向最大强度投影](docs/assets/reconstruction_preview.png)

*附带 pose000 示例的正交最大强度投影。三个方向使用同一完整体积最大值线性显示，没有额外显示平滑或百分位截断。*

初始点云包含 1,000,000 个随机球，细分后有 1,420,000 个候选球，最终导出 **592,864 个非零 Gaussian**。点云、400³ 体积和预测 RF 位于 [`examples/pose000/final/`](examples/pose000/final/)。

该示例使用全部 1024 个探头完成最终训练和响应校准，其 RF MSE `199.567426` 是拟合内指标。部分细分支连续性仍弱于 PA 结构初始化的比较结果。完整对照见 [结果与限制](docs/RESULTS.md)；示例图像不作为训练输入。

## Guidance / 使用前说明

运行自己的数据只需提供两个输入文件：**有符号的光声 RF 信号**和**对应的探头坐标**。程序会自动准备归一化信号、探头划分和 metadata，无需手工生成这些中间文件。

| 输入 | 默认数组形状 | 约定 |
|---|---|---|
| RF 信号 | `[N, T]`，探头 × 时间点 | 第 `i` 行对应第 `i` 个探头，保留信号正负号 |
| 探头坐标 | `[N, 3]`，探头 × XYZ | 默认单位为米，可显式指定 `mm` 或 `cm` |

支持 **NPY、NPZ、TXT、CSV、MAT（非 v7.3/HDF5）**。NPZ/MAT 有多个变量时，通过 `input.signal_key` 和 `input.sensors_key` 指定数组名。TXT 默认按空白分隔，CSV 默认按逗号分隔；标题行可用 `input.signal_skiprows` / `input.sensors_skiprows` 跳过。

若 RF 为 `[T,N]`，设置 `input.signal_layout="time_sensor"`；若坐标为 `[3,N]`，设置 `input.sensor_layout="xyz_sensor"`。这些选项仅处理转置和单位换算，不会重新排列通道或配准坐标。

当前重建范围为 XYZ 各 **±20 mm**，导出网格为 **400³、0.1 mm**；时间零点要求 **`t0_seconds=0`**。输入应已完成所需的采集预处理和时间对齐，本项目不会自动恢复 ADC 处理链、滤波、取绝对值或调整时间零点。附带数据也属于已处理 RF。

## Installation / 安装

获取本项目目录后，在终端进入该目录并安装环境：

```bash
cd SlingBAG_3D-PanoPACT

python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

python scripts/verify_package.py
python scripts/check_environment.py --cuda
```

已验证环境：Linux、Python 3.10、PyTorch 2.5.1+cu121、Triton 3.1.0、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.10.8、NVIDIA A100 40 GB。PyTorch 安装方式参见[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v251)。

GPU 重建还需要可用的 **CUDA Toolkit / `nvcc`、C++ 编译器和 Ninja**；PyTorch 的 CUDA 运行时包不替代本地编译工具链。首次运行会编译扩展。Triton 用于快速体素化，缺少时有较慢的 PyTorch 后备实现。CPU 查看数据和预览无需 GPU。

`run.sh` 默认使用当前环境的 `python`。已有环境可通过 `SLINGBAG_PYTHON=/path/to/python ./run.sh ...` 指定解释器。以下命令均在项目目录执行；配置中的相对路径也以项目根目录为基准。

## Usage / 使用方法

### 1. 查看附带结果

```bash
./run.sh scripts/preview.py
```

读取附带的 `examples/pose000/final/volume_zyx.npy`，在 `outputs/preview.png` 生成三方向投影，全程使用 CPU。

### 2. 重建附带的这一帧

先用短流程确认环境和各阶段能正常衔接：

```bash
./run.sh reconstruct.py --config configs/smoke.json
```

该配置只有 4096 个初始球、每阶段 3 步，**仅检查执行流程，不代表重建质量**。完整重建使用：

```bash
./run.sh reconstruct.py --config configs/pose000.json
```

程序重新读取本帧原幅值 RF，计算增益并随机初始化点云。准备数据保存到 `outputs/pose000/data/`，最终重建保存到 `outputs/pose000/final/`。默认流程为：

```text
读取 RF / 坐标 → 自动归一化与探头划分
 → 随机初始化 → coarse → 守恒细分 → fine → 细球变窄
                                      ├→ narrow → FIR 拟合与点更新诊断 → validated
                                      └→ 全探头 refit_all → 导出 → 最终 FIR 校准 → final
```

新运行不会覆盖原输入或附带示例。完整运行建议预留约 5 GB 空间；附带数据及示例约 361 MiB。CUDA 原子累加可能产生末位差异，同一随机种子不保证逐位重现示例。

### 3. 运行自己的数据：直接指定参数

以下例子假设 RF 为 `[N,T]`、坐标为以米表示的 `[N,3]`。将两个路径以及声速、采样频率改为实际采集值：

```bash
./run.sh reconstruct.py \
  --signal /path/to/signal.npy \
  --sensors /path/to/sensors.npy \
  --sound-speed 1480 \
  --sampling-frequency 20000000 \
  --samples auto \
  --coordinate-unit m \
  --output outputs/new_frame
```

**声速用 m/s，采样频率用 Hz。** `20000000` 表示 20 MHz；`--samples auto` 自动识别时间点数。若坐标为毫米，改用 `--coordinate-unit mm`。探头数量由信号和坐标共同确定。

| 要调整的内容 | JSON 配置字段 | 命令行参数 | 默认值 / 单位 |
|---|---|---|---|
| 信号文件 | `input.signal` | `--signal` | `data/pose000/signal_original.npy` |
| 探头坐标文件 | `input.sensors` | `--sensors` | `data/pose000/sensor_locations.npy` |
| 声速 | `acquisition.sound_speed_m_s` | `--sound-speed` | `1500`，m/s |
| 采样频率 | `acquisition.sampling_frequency_hz` | `--sampling-frequency` | `40000000`，Hz（40 MHz） |
| 时间采样点数 | `acquisition.samples` | `--samples` | JSON `null` / CLI `auto`：自动读取 |
| 坐标单位 | `input.coordinate_unit` | `--coordinate-unit` | `m`；可选 `m/mm/cm` |
| RF 轴序 | `input.signal_layout` | `--signal-layout` | `sensor_time`；也可选 `time_sensor` |
| 归一化目标 RMS | `normalization.target_rms` | `--target-rms` | `15.941105800921543` |
| 输出目录 | `default_output` | `--output` | `outputs/pose000` |

命令行参数会覆盖相应配置值。`samples` 填整数时仅校验输入长度，**不会裁剪、补零或重采样**；采样间隔自动计算为 `dt=1/fs`。这些参数会实际传入训练、导出和响应校准，而不仅是写入说明文件。

### 4. 保存自己的配置，先检查数据再训练

复制完整配置：

```bash
cp configs/pose000.json configs/my_data.json
```

在新文件中修改以下部分，**保留其余训练、细分和导出设置**。下面是配置节选，不是完整运行配置：

```json
{
  "input": {
    "signal": "/path/to/signal.npy",
    "sensors": "/path/to/sensors.npy",
    "signal_key": null,
    "sensors_key": null,
    "signal_layout": "sensor_time",
    "sensor_layout": "sensor_xyz",
    "coordinate_unit": "m",
    "frame": "my_data"
  },
  "acquisition": {
    "sound_speed_m_s": 1480.0,
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

`frame` 为描述性标签，可改名或删除；删除后采用信号文件名。`sensor_split.file=null` 自动生成探头划分，默认约 87.5% 训练、12.5% 验证，至少需要两个探头。已有划分可填写含 `train`、`validation` 数组的 NPZ 路径。

然后执行同一份新配置：

```bash
# 只准备数据、计算增益，不开始 GPU 训练
./run.sh reconstruct.py --config configs/my_data.json \
  --output outputs/my_data --prepare-only

# 检查形状、采样参数、单位、实际 RMS、增益和探头划分
./run.sh scripts/check_environment.py --data-dir outputs/my_data/data

# 保持相同配置，继续完整重建
./run.sh reconstruct.py --config configs/my_data.json \
  --output outputs/my_data --resume
```

`--resume` 会核对配置、源文件和已完成产物，跳过完成的阶段。中断的训练阶段从该阶段初值重跑，不恢复中间 Adam 状态。若修改输入或参数，请使用新的输出目录；续跑时必须保留原来的配置和命令行覆盖参数。

### 5. 自动计算归一化增益

每次新运行都根据**当前这一帧**的全部 RF 样本计算：

```text
measured_rms     = sqrt(mean(signal_original ** 2))
global_gain      = normalization.target_rms / measured_rms
signal_normalized = signal_original * global_gain
```

RMS 使用平方均值，不先减去均值。所有探头共用同一个正增益，保留信号正负号及探头间相对幅值，不做逐通道归一化。默认目标约 `15.9411`，沿用仿真的数值幅值尺度，不代表绝对声压标定。

**不需要手动计算增益，也不要复用上一帧的增益。** 程序在 `outputs/<run>/data/metadata.json` 中记录 `global_gain`、输入 RMS、目标 RMS、实际输出 RMS 和采集参数。训练读取已准备好的信号，不重复缩放；同一运行恢复时校验并复用该快照。原幅值预测可由归一化预测除以这一帧的增益得到。

### 6. 查看输出和导出体积

对上面的 `my_data` 运行：

```bash
./run.sh scripts/preview.py \
  --volume outputs/my_data/final/volume_zyx.npy \
  --metadata outputs/my_data/data/metadata.json \
  --output outputs/my_data/preview.png
```

| 路径（相对于运行目录） | 内容 |
|---|---|
| `data/` | 原幅值与归一化 RF、米制坐标、metadata 和探头划分 |
| `final/checkpoint.pt` | 球心 `xyz`、峰值 `pressure`、标准差 `radius`（σ）及配置 |
| `final/point_cloud.ply` | 非零 Gaussian 点云；XYZ 和 σ 均为米 |
| `final/volume_zyx.npy` | `float32[400,400,400]`，轴序 **ZYX**，体素间距 0.1 mm |
| `final/acoustic_signal_normalized.npy` | 未卷积响应的声学预测 |
| `final/predicted_signal_normalized.npy` | 共享 FIR 校准后的预测 |
| `final/response_kernel.npy` | 共享经验响应核，默认 41 tap |
| `final/metrics.json` / `final/response_calibration.json` | 误差、评价范围及校准记录 |
| `logs/`、`config.json`、`run.json` | 阶段日志、有效配置和完成记录 |

默认不重复输出大型 MAT 体积。在自己的配置中将 `export_mat` 改为 `true` 后启动新运行，可同时生成 `volume_zyx.mat`。FIR 只改变 RF 预测，不改变点云或体积。

## Repository structure / 目录结构

```text
SlingBAG_3D-PanoPACT/
├── README.md / README.en.md
├── reconstruct.py          # 一条命令执行完整流程
├── run.sh                  # 使用当前 Python 环境
├── requirements.txt
├── configs/                # 完整配置和短流程检查配置
├── slingbag/               # 数据准备、CUDA 前向、训练、细分、响应与导出
├── scripts/                # 数据与环境检查、预览和数值检查
├── data/pose000/            # 附带这一帧的信号、坐标和元数据
├── examples/pose000/final/  # 已完成的示例点云与体积
├── tests/                  # CPU 数值、数据和配置测试
├── docs/                   # 详细方法、数据和结果说明
├── provenance/             # 来源、文件哈希和验证记录
└── outputs/                # 新运行数据与结果，Git 忽略
```

## Tips / 注意事项

- 输入坐标须与 RF 通道顺序、重建坐标系对应。单位选项不能修正通道错位或空间配准。
- 目前 `t0=0`、±20 mm 视野和 400³ 网格固定；声速、采样频率、时间点数及探头数量已可配置。不同采集条件下仍需评估重建质量。
- 采样频率改变后，相同 FIR tap 数对应的物理时间宽度也会改变；响应半宽可在 `response_fit.half_width` 中调整，须小于 RF 时间点数。
- 显存不足可先用 smoke 配置定位问题。更改 `shared_training.batch_sensors` 或先验网格会改变优化行为；CIC 方差补偿要求 `σ > 网格间距/√6`。
- 没有 `nvcc` 时需安装 CUDA Toolkit 并配置 `CUDA_HOME` / `PATH`。`nvidia-smi` 可运行不代表编译器可用。可通过 `SLINGBAG_CUDA_ARCH` 指定编译架构。
- 从 Git LFS 仓库获取数据时，先运行 `git lfs pull`。建议在修改默认文件前运行完整性检查，然后复制配置到新文件中修改；主动修改已记录的源码或配置会使原分发哈希变化。
- 旧 `schema_version=1` 的 `data_dir` 配置仍支持读取已准备数据，但不会自动重新归一化。新数据请使用当前的 `schema_version=2` 配置。

```bash
python scripts/verify_package.py
./run.sh -m unittest discover -s tests -v
./run.sh scripts/check_cuda.py --run
```

当前代码已通过 46 项 CPU 测试，以及原规格和不同采集规格的两组 GPU 短流程。记录见 [验证说明](docs/VALIDATION.md)。短流程验证执行与数值正确性，不替代完整重建的质量评估。

更多细节：[新数据格式](docs/NEW_DATA.md) · [pose000 数据说明](docs/DATA.md) · [算法说明](docs/METHOD.md) · [结果与限制](docs/RESULTS.md)。

## BibTeX / 原始 SlingBAG 论文引用

以下条目沿用[原始 SlingBAG 仓库](https://github.com/JaegerCQ/SlingBAG)。本项目是基于其思路的变体，附带结果不应归为未经修改的原始算法结果。

```bibtex
@article{li2025slingbag,
  title={SlingBAG: point cloud-based iterative algorithm for large-scale 3D photoacoustic imaging},
  author={Li, Shuang and Wang, Yibing and Gao, Jian and Kim, Chulhong and Choi, Seongwook and Zhang, Yu and Chen, Qian and Yao, Yao and Li, Changhui},
  journal={Nature Communications},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```

## License / 许可

保留原始代码的 [MIT 许可](LICENSE)。附带实验数据没有单独提供再分发许可文本，代码许可不自动说明数据授权范围。`.gitignore` 和 `.gitattributes` 已为生成结果及 Git LFS 二进制文件做好设置；当前目录为本地项目，未创建远程仓库。
