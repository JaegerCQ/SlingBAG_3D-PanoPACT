# pose000 数据与数值约定

仓库包含一帧 `group03_pose000`，足够离线检查数据和查看预生成结果。训练和导出不需要访问此前项目目录。文件的 SHA256、历史来源标识及复制方式见 [`provenance/portable_manifest.json`](../provenance/portable_manifest.json)。`historical:` 开头的标识只描述历史来源，不是可执行路径。

## 文件

| 文件 | 内容与形状 |
|---|---|
| `data/pose000/signal_original.npy` | 已处理、未应用本项目幅值缩放的有符号 RF，`[1024,4096]` |
| `data/pose000/signal_normalized.npy` | 对上述 RF 乘同一个全局标量，`[1024,4096]` |
| `data/pose000/sensor_locations.npy` | 与 RF 行顺序一致的探头 XYZ 坐标，`[1024,3]`，单位米 |
| `data/pose000/split.npz` | `train` 为 896 个探头索引，`validation` 为 128 个索引 |
| `data/pose000/split.json` | 同一划分的可读记录，随机种子 `20261008` |
| `data/pose000/metadata.json` | 采样、坐标、归一化和原始来源摘要 |

`signal_original.npy` **不是原始 ADC 数据**。它来自上游仓库提供的 `processed_signal_group03_pose000.txt`；上游未提供完整的 ADC 到该文件的处理链。这里不猜测上游滤波、时间零点校正或硬件响应，也不追加取绝对值、翻转极性、逐通道归一化、抽取或额外除以 50。

## 采样、坐标与幅值

- 时间间隔为 `25 ns`，采样率 `40 MHz`，4096 个样本，`t[n]=n×25 ns`；最后一个样本为 `102.375 μs`。
- 声速为 `1500 m/s`。第 `i` 行 RF 对应第 `i` 个传感器坐标，pose000 不额外应用坐标变换。
- 重建范围为 XYZ 各 `[-0.02,0.02] m`。示例体积使用 `400³` 个 `0.1 mm` 体素，数组轴序为 **ZYX**；首个体素中心 XYZ 均为 `−0.01995 m`。
- `signal_normalized = signal_original × 0.2612356995077034`。这是保留所有传感器相对幅值的单一标量。该帧全部探头的 RF RMS 从约 `61.021927` 变为 `15.941107`，与历史仿真示例的整体 RMS 接近。仿真文件不是本仓库运行依赖。
- 该归一化仅定义数值幅度尺度，未建立帕斯卡或其他绝对物理压力标定。它使用整帧 RMS，不能声称为严格只使用训练子集的预处理。

## 划分的用途

896/128 划分用于算法开发：896 个探头拟合点参数和候选共享 FIR，128 个探头选择迭代、正则权重和 FIR ridge。该 128 子集已经用于调参，**不是独立测试集**。

预生成的 `examples/pose000/final` 使用全部 1024 探头完成点模型复拟，再用全部 1024 探头估计固定 ridge=10 的共享响应。因此其两个原划分子集都是训练和校准数据，不能再称为验证集。JSON 中保留 `original_validation_subset` 键以兼容历史格式，其 `role` 明确说明这一点。

## 预生成点云和体积

`examples/pose000/final/checkpoint.pt` 包含 `xyz`、`pressure`、`radius` 以及配置和记录。这里沿用字段名 `radius`，其数值实际是各向同性 Gaussian 的标准差 **σ，单位米**。`pressure` 是每个 Gaussian 的峰值系数，物理场为各球 `P exp(−‖x−μ‖²/(2σ²))` 的和；单球积分质量为 `(2π)^(3/2) P σ³`。

示例包含与 checkpoint 一致的 PLY、`volume_zyx.npy`、未校准的 `acoustic_signal_normalized.npy`、校准后的 `predicted_signal_normalized.npy` 和 `response_kernel.npy`。后者是 41 tap 的两侧共享卷积核，lag 为 `−20…20`，正 lag 表示延迟预测。原幅度预测可由 `predicted_signal_normalized / global_gain` 得到；重复的 MAT 体积未打包，可由导出程序生成。

包内没有 PA/DAS/hybrid 参考体积、训练缓存或全部历史中间结果。比较图片仅用于展示历史结果，不作为初始化、正则目标或训练输入。

## 溯源与完整性

清单中每项包含仓库相对 `path`、`size_bytes`、当前 `sha256`、`transform` 和可选历史 `source`。`byte_copy` 项与源文件字节一致；`metadata_only` 项只调整元数据或历史路径；`derived_document` 项为整理后的说明或摘要。

checkpoint 重新序列化后文件 SHA256 会变化，因此清单另记录 `xyz`、`pressure`、`radius` 的源/新 tensor SHA256、形状、dtype 和 `bitwise_equal`。原始路径已移除，历史文件哈希仍保留作溯源；检查本包内容时应使用 portable manifest，不能把历史运行报告中的旧 checkpoint/JSON 哈希当作当前文件哈希。
