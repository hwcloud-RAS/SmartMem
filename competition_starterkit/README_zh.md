# SmartMem Competition Baseline

[View in English](README.md)

## 概述

本程序是 [The Web Conference 2025 Competition: SmartMem (Memory Failure Prediction for Cloud Service Reliability)](https://www.codabench.org/competitions/3586/)
的 baseline，主要通过对内存日志数据进行分析，提取时间、空间和奇偶校验等特征，并使用 LightGBM
模型进行训练和预测。程序支持多进程处理，能够高效地处理大规模数据。

竞赛主页：[https://hwcloud-ras.github.io/SmartMem.github.io/](https://hwcloud-ras.github.io/SmartMem.github.io/)

## 功能模块

1. **配置管理 (`Config` 类)**
    - 用于管理程序的配置信息，包括数据路径、时间窗口大小、特征提取间隔等。
    - 支持多进程处理，可配置并行 worker 数量。

2. **特征提取 (`FeatureFactory` 类)**
    - 从原始日志数据中提取时间、空间和奇偶校验特征。
    - 支持多进程并行处理，能够高效地处理大量 SN 文件。
    - 生成的特征数据保存为 `.feather` 格式，便于后续处理。

3. **数据生成 (`DataGenerator` 类及其子类)**
    - **正样本数据生成 (`PositiveDataGenerator` 类)**: 结合维修单数据，从故障 SN 中提取正样本数据。
    - **负样本数据生成 (`NegativeDataGenerator` 类)**: 从未发生故障的 SN 中提取负样本数据。
    - **测试数据生成 (`TestDataGenerator` 类)**: 生成测试数据，用于模型预测。

4. **模型训练与预测 (`MFPmodel` 类)**
    - 使用 LightGBM 模型进行训练和预测。
    - 支持加载训练数据、训练模型、预测测试数据等功能。
    - 按照竞赛要求，将预测结果保存为 CSV 文件。

## 使用说明

### 1. 环境准备

- 确保已安装以下 Python 库：
    - `feather-format`
    - `joblib`
    - `lightgbm`
    - `numpy`
    - `pandas`
    - `pyarrow`
    - `scipy`
    - `tqdm`
- 推荐使用 Python 3.8 及以上版本，可以通过以下命令安装依赖库：
    ```bash
    pip install -r requirements.txt
    ```

### 2. 下载数据集

- 我们提供了两种格式的数据集：`csv` 和 `feather`。
    - **CSV 格式**：解压后约 130G，适合需要直接查看或处理原始文本数据的场景。
    - **Feather 格式**：解压后约 40G，适合需要高效读取和处理数据的场景，性能优于 CSV 格式。
- 请根据需求选择合适的数据集格式，并确保下载后解压到正确的路径。

### 3. 配置文件

- 在 `Config` 类中配置数据路径等参数。
- 确保数据路径正确，并根据使用的数据集格式设置 `DATA_SUFFIX`：
    - 如果使用 `csv` 文件，则设置 `DATA_SUFFIX` 为 `csv`。
    - 如果使用 `feather` 文件，则设置 `DATA_SUFFIX` 为 `feather`。

### 4. 运行程序

- 直接运行 `baseline.py` 脚本，程序将依次执行以下步骤：
    1. 初始化配置。
    2. 提取特征并保存。
    3. 生成正样本、负样本和测试数据。
    4. 训练模型并进行预测。
    5. 将预测结果保存为 `submission.csv` 文件。

### 5. 输出文件

- **特征文件**: 保存在 `feature_path` 指定的路径下，格式为 `.feather`。
- **训练数据**: 正样本和负样本数据分别保存在 `train_data_path` 指定的路径下，格式为 `.feather`。
- **测试数据**: 保存在 `test_data_path` 指定的路径下，格式为 `.feather`。
- **预测结果**: 保存为 `submission.csv` 文件，包含 SN 名称、预测时间戳和 SN 类型。

### 6. 提交说明

请将生成的 **submission.csv** 文件 **压缩为 zip** 后提交到 [SmartMem 竞赛页面](https://www.codabench.org/competitions/3586/)。
