# 打包与运行验证

当前版本新增了 schema 2 数据准备与可配置采集参数。对应测试见 [`configurability_validation.json`](../provenance/configurability_validation.json)，使用方法见 [NEW_DATA.md](NEW_DATA.md)。下文记录初次打包时的验证；历史记录中的代码哈希对应当时的版本，不应当作当前代码哈希。当前分发文件以 source_manifest 为准。

schema 2 更新已通过 46 项 CPU 测试，并在 A100 上分别完成两组 14 阶段短流程：原规格 1024×4096 / 40 MHz / 1500 m/s，以及合成数据 24×1024 / 20 MHz / 1480 m/s。合成测试还覆盖 NPZ 转置输入、CSV 毫米坐标、自动划分、15 tap FIR、自定义 ridge，以及仅准备数据后恢复训练。72 个 RF 样本对独立 FP64 公式的相对 L2 误差为 `2.1449e−6`。

原帧重新计算的增益为 `0.2612356995077035`，与历史值 `0.2612356995077034` 一致到浮点精度；训练/验证索引逐位相同。归一化乘法改用 FP64 后存回 FP32，与历史已存信号的相对 L2 差异为 `6.9014e−8`。原 data/examples 文件保持逐字节不变，没有将短流程产物替换为示例结果。

本次整理保留此前最终重建的数值结果，并对独立目录的代码、路径和执行流程进行检查。没有为这次打包重新运行完整百万球训练；附带结果来自已完成的原随机初始化实验。新目录执行的 smoke 只有 4096 个初始球、每个训练阶段 3 步，不能用于图像质量评价。

## 检查范围

| 检查 | 结果 |
|---|---|
| 本帧 RF / 坐标 / 划分 / 示例 NPY 和 PLY | 原新文件 SHA256 一致 |
| checkpoint XYZ、压力、σ | 清理历史路径后 tensor 逐位一致，共 592,864 个非零球 |
| 数据归一化与评价 | 全局增益关系、896/128 划分、最终 RF MSE 复算通过 |
| 配置迁移 | 五个训练阶段的超参数与原选定流程逐项一致 |
| 科学计算核心 | CUDA 源文件字节一致；前向、体素化、先验及 FIR 核心数学未改动 |
| CPU 数学检查 | 连续前向梯度、空源、网格 XYZ / 压力梯度、FIR 稠密参考通过 |
| CUDA 检查 | 前向、XYZ / 压力 / σ 一阶梯度、固定 XYZ 分支、空源及体素化通过 |
| 完整执行流程 | 13 个 smoke 阶段完成，含从随机开始、位置更新、细分、变窄、FIR、全探头拟合、导出和预览 |
| 完成阶段恢复 | `--resume` 核对产物并跳过全部已完成阶段 |
| 校准备份恢复 | 6 个 CPU 场景检查：允许补齐一致的部分备份，拒绝冲突或已校准输入 |
| 目录迁移 | 复制完整代码、数据和示例后，从 `/tmp` 启动；dry-run、CPU 检查和清单验证通过 |

环境检查只检查可用性、基本 CUDA 算术与数据一致性；实际扩展编译和梯度正确性由独立的 CUDA 数值检查及 smoke 覆盖。

## CUDA 对 FP64 的数值检查

使用 3 个探头、4 个 Gaussian、512 个时间点的合成场景，FP64 参考输入由 CUDA 实际 FP32 输入精确提升而来。对未截断与 ±6σ 两种情况测试相同的上游梯度；XYZ/σ 梯度按毫米参数缩放。

| 量 | 相对 L2 误差范围 |
|---|---:|
| RF 前向 | 约 `2.32e−6` |
| XYZ 梯度 | 约 `2.87e−6` |
| 压力梯度 | `3.45e−6` 至 `3.50e−6` |
| σ 梯度 | `7.46e−6` 至 `7.53e−6` |
| 小体积 Gaussian 导出 | 约 `1.65e−7` |

检查同时使用逐元素绝对/相对容差。详细参数和实测值见 [`provenance/cuda_numerics.json`](../provenance/cuda_numerics.json)。这是一阶小样本正确性检查，不是吞吐量测试，也不替代实测重建质量评估。

## 实际测试环境

使用 Python 3.10、PyTorch 2.5.1+cu121、Triton 3.1.0、NumPy 1.26.4、SciPy 1.15.3、Matplotlib 3.10.8 和 A100 40 GB。实际本地 nvcc 为 CUDA Toolkit 12.2，系统 Ninja 为 1.10.1；环境脚本提示了 Toolkit 与 PyTorch CUDA 12.1 的次版本差异，该组合已成功编译并完成测试。requirements 为新安装指定 Ninja ≥1.11，不表示实测环境的每个包都来自一次全新 requirements 安装。

尚未验证全新机器的依赖安装、其他 GPU 或完整训练的跨机器质量重现。GPU 原子累加顺序允许末位差异，不能保证重新训练得到逐位一致的点云。

## 记录与复核

数据和预生成示例的来源及哈希在 [`portable_manifest.json`](../provenance/portable_manifest.json)，数值保持记录在 [`payload_validation.json`](../provenance/payload_validation.json)。其余代码、配置和说明的分发哈希在 [`source_manifest.json`](../provenance/source_manifest.json)。打包测试摘要在 [`packaging_validation.json`](../provenance/packaging_validation.json)。smoke 的大体积中间文件与机器编译缓存不作为分发内容保留。

```bash
python scripts/verify_package.py
python scripts/check_environment.py --cuda
./run.sh -m unittest discover -s tests -v
./run.sh scripts/check_cuda.py --run --output outputs/validation/cuda_numerics.json
./run.sh reconstruct.py --config configs/smoke.json
./run.sh reconstruct.py --config configs/smoke.json --resume
```

打包的精简示例适合预览和重新导出。它未携带最终校准的全部声学备份，不能在其目录内直接重复校准；完整新运行会生成这些备份，并支持从声学模型幂等地重新校准。
