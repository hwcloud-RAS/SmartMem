# SmartMem 竞赛：入门工具包与基线

[View in English](README.md)

## 概述

本仓库提供了 [The Web Conference 2025 Competition: SmartMem (Memory Failure Prediction for Cloud Service Reliability)](https://www.codabench.org/competitions/3586/) 的资源，包括：

1. **入门工具包**：一个 Jupyter notebook，提供了数据处理、特征工程和构建机器学习模型的分步指南。旨在帮助参赛者快速搭建工作流程并优化预测性能。
2. **基线实现**：一个强大的管道，用于特征提取、数据准备和模型训练，能够高效处理大规模数据。

竞赛主页：[https://hwcloud-ras.github.io/SmartMem.github.io/](https://hwcloud-ras.github.io/SmartMem.github.io/)

## 入门工具包

入门工具包是一个 Jupyter notebook，旨在引导参赛者完成竞赛工作流程，其特点包括：

- **分步工作流程**：涵盖数据加载、预处理、特征工程、模型训练和评估。
- **预配置工具**：包括 XGBoost 和 scikit-learn 等常用库的预加载配置。
- **可重复的结果**：提供了预分割的数据集，确保训练和测试结果的一致性。
- **可视化与指标**：生成性能指标（如混淆矩阵和特征重要性图）的工具。
- **可扩展性**：易于修改以进行高级特征工程或试验替代模型。
- **最佳实践**：强调正确的数据处理、评估技术和模块化代码设计。

## 基线实现

基线程序分析内存日志数据，提取时间、空间和奇偶校验特征，并使用 LightGBM 模型进行训练和预测。主要特点包括：

1. **配置管理 (`Config` 类)**
    - 管理程序的配置信息，包括数据路径、时间窗口大小、特征提取间隔等。
    - 支持多进程并行处理，可配置并行 worker 数量。

2. **特征提取 (`FeatureFactory` 类)**
    - 从原始日志数据中提取时间、空间和奇偶校验特征。
    - 支持多进程高效处理大量数据文件。
    - 生成的特征保存为 `.feather` 格式，便于后续处理。

3. **数据生成 (`DataGenerator` 类及其子类)**
    - **正样本生成 (`PositiveDataGenerator` 类)**：结合维修单数据，从故障 SN 中提取正样本。
    - **负样本生成 (`NegativeDataGenerator` 类)**：从未发生故障的 SN 中提取负样本。
    - **测试数据生成 (`TestDataGenerator` 类)**：生成测试数据，用于模型预测。

4. **模型训练与预测 (`MFPmodel` 类)**
    - 使用 LightGBM 进行训练和预测。
    - 支持加载训练数据、训练模型和预测测试数据。
    - 按照竞赛要求保存预测结果为 CSV 文件。

## 使用说明

### 1. 环境准备

- 确保已安装 Python 3.8 或更高版本。
- 安装所需的 Python 库：
  ```bash
  pip install -r requirements.txt
  ```

### 2. 数据集准备

- 数据集有两种格式：`csv` 和 `feather`：
  - **CSV 格式**：解压后约 130G，适合需要直接访问或处理原始文本数据的场景。
  - **Feather 格式**：解压后约 40G，适合需要高效数据处理的场景，性能优于 CSV 格式。
- 下载数据集后解压到正确路径，并在配置文件中更新路径。

### 3. 配置

- 在 `Config` 类中配置数据路径及其他参数。
- 根据使用的数据集格式设置 `DATA_SUFFIX`：
  - 对于 `csv` 文件，将 `DATA_SUFFIX` 设置为 `csv`。
  - 对于 `feather` 文件，将 `DATA_SUFFIX` 设置为 `feather`。

### 4. 运行基线程序

运行 `baseline.py` 脚本以完成以下任务：

1. 初始化配置。
2. 提取特征。
3. 生成正负样本和测试数据。
4. 训练模型并预测。
5. 将预测结果保存为 `submission.csv` 文件。

### 5. 使用入门工具包

打开入门 notebook：
```bash
jupyter notebook starterkit_notebook.ipynb
```
按照顺序运行各单元格：
- 加载并预处理数据集。
- 进行特征工程。
- 训练和评估机器学习模型。

### 6. 输出文件

- **特征文件**：保存在 `feature_path` 指定的路径下，格式为 `.feather`。
- **训练数据**：正样本和负样本数据保存在 `train_data_path` 指定的路径下，格式为 `.feather`。
- **测试数据**：保存在 `test_data_path` 指定的路径下，格式为 `.feather`。
- **预测结果**：保存在 `submission.csv` 文件中，包括 SN 名称、预测时间戳和 SN 类型。

### 7. 提交说明

将生成的 `submission.csv` 文件压缩为 zip 格式，并提交到 [SmartMem 竞赛页面](https://www.codabench.org/competitions/3586/)。

## 注意事项

- 根据需要扩展入门工具包或基线程序以提高性能。
- 根据本地环境更新 `Config` 类中的路径和配置。

## 许可证

本项目基于 MIT 许可证发布。详细信息请参阅 LICENSE 文件。

---
欢迎通过提交 PR 或问题来贡献代码！
