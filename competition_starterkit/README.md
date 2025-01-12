# SmartMem Competition Baseline

[查看中文版 / View in Chinese](README_zh.md)

## Overview

This program is the baseline for [The Web Conference 2025 Competition: SmartMem (Memory Failure Prediction for Cloud Service Reliability)](https://www.codabench.org/competitions/3586/). It primarily analyzes memory log data, extracts temporal, spatial, and parity features, and uses the LightGBM model for training and prediction. The program supports multi-processing, enabling efficient handling of large-scale data.  

Competition homepage: [https://hwcloud-ras.github.io/SmartMem.github.io/](https://hwcloud-ras.github.io/SmartMem.github.io/)

## Functional Modules

1. **Configuration Management (`Config` Class)**
    - Manages program configuration information, including data paths, time window sizes, feature extraction intervals, etc.
    - Supports multi-processing and allows configuration of the number of parallel workers.

2. **Feature Extraction (`FeatureFactory` Class)**
    - Extracts temporal, spatial, and parity features from raw log data.
    - Supports multi-processing for efficient handling of a large number of SN files.
    - Saves extracted features in `.feather` format for subsequent processing.

3. **Data Generation (`DataGenerator` Class and Subclasses)**
    - **Positive Sample Generation (`PositiveDataGenerator` Class)**: Extracts positive samples from SNs with failures, combined with maintenance ticket data.
    - **Negative Sample Generation (`NegativeDataGenerator` Class)**: Extracts negative samples from SNs without failures.
    - **Test Data Generation (`TestDataGenerator` Class)**: Generates test data for model prediction.

4. **Model Training and Prediction (`MFPmodel` Class)**
    - Uses the LightGBM model for training and prediction.
    - Supports loading training data, training the model, and predicting test data.
    - Saves prediction results in a CSV file as required by the competition.

## Usage Instructions

### 1. Environment Setup

- Ensure the following Python libraries are installed:
    - `feather-format`
    - `joblib`
    - `lightgbm`
    - `numpy`
    - `pandas`
    - `pyarrow`
    - `scipy`
    - `tqdm`
- Python 3.8 or higher is recommended. Install the required libraries using the following command:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Download Dataset

- We provide datasets in two formats: `csv` and `feather`.
    - **CSV Format**: Approximately 130G when decompressed, suitable for scenarios requiring direct access to or processing of raw text data.
    - **Feather Format**: Approximately 40G when decompressed, suitable for scenarios requiring efficient data reading and processing, with better performance than CSV.
- Choose the appropriate dataset format based on your needs and ensure it is decompressed to the correct path.

### 3. Configuration

- Configure data paths and other parameters in the `Config` class.
- Ensure the data paths are correct and set `DATA_SUFFIX` based on the dataset format used:
    - If using `csv` files, set `DATA_SUFFIX` to `csv`.
    - If using `feather` files, set `DATA_SUFFIX` to `feather`.

### 4. Run the Program

- Execute the `baseline.py` script directly. The program will perform the following steps in sequence:
    1. Initialize configuration.
    2. Extract and save features.
    3. Generate positive samples, negative samples, and test data.
    4. Train the model and perform predictions.
    5. Save the prediction results to a `submission.csv` file.

### 5. Output Files

- **Feature Files**: Saved in the path specified by `feature_path`, in `.feather` format.
- **Training Data**: Positive and negative samples are saved in the path specified by `train_data_path`, in `.feather` format.
- **Test Data**: Saved in the path specified by `test_data_path`, in `.feather` format.
- **Prediction Results**: Saved in a `submission.csv` file, containing SN names, prediction timestamps, and SN types.

### 6. Submission Instructions

Compress the generated **submission.csv** file into a **zip** file and submit it to the [SmartMem Competition Page](https://www.codabench.org/competitions/3586/).