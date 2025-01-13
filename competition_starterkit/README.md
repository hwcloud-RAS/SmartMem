# SmartMem Competition: Starter Kit and Baseline

[查看中文版 / View in Chinese](README_zh.md)

## Overview

This repository contains resources for the [The Web Conference 2025 Competition: SmartMem (Memory Failure Prediction for Cloud Service Reliability)](https://www.codabench.org/competitions/3586/). It includes:

1. **Starter Kit**: A Jupyter notebook providing a step-by-step guide for data processing, feature engineering, and building machine learning models. The notebook is designed to help participants quickly set up a working pipeline and improve predictive performance.
2. **Baseline Implementation**: A robust pipeline for feature extraction, data preparation, and model training, designed for efficient handling of large-scale data.

Competition homepage: [https://hwcloud-ras.github.io/SmartMem.github.io/](https://hwcloud-ras.github.io/SmartMem.github.io/)

## Starter Kit

The starter kit is a Jupyter notebook designed to guide participants through the competition workflow. It features:

- **Step-by-Step Workflow**: Covers data loading, preprocessing, feature engineering, model training, and evaluation.
- **Preconfigured Tools**: Includes preloaded configurations for libraries like XGBoost and scikit-learn.
- **Reproducible Results**: Provides pre-split datasets for consistent training and testing.
- **Visualization and Metrics**: Tools for generating performance metrics such as confusion matrices and feature importance plots.
- **Extensibility**: Easily modifiable for advanced feature engineering and experimentation with alternative models.
- **Best Practices**: Emphasizes proper data handling, evaluation techniques, and modular code design.

## Baseline Implementation

The baseline program analyzes memory log data, extracts temporal, spatial, and parity features, and uses the LightGBM model for training and prediction. Key features include:

1. **Configuration Management (`Config` Class)**
    - Manages program configuration information, including data paths, time window sizes, feature extraction intervals, etc.
    - Supports multi-processing and allows configuration of the number of parallel workers.

2. **Feature Extraction (`FeatureFactory` Class)**
    - Extracts temporal, spatial, and parity features from raw log data.
    - Supports multi-processing for efficient handling of large data files.
    - Saves extracted features in `.feather` format for subsequent processing.

3. **Data Generation (`DataGenerator` Class and Subclasses)**
    - **Positive Sample Generation (`PositiveDataGenerator` Class)**: Extracts positive samples from SNs with failures, combined with maintenance ticket data.
    - **Negative Sample Generation (`NegativeDataGenerator` Class)**: Extracts negative samples from SNs without failures.
    - **Test Data Generation (`TestDataGenerator` Class)**: Generates test data for model prediction.

4. **Model Training and Prediction (`MFPmodel` Class)**
    - Uses LightGBM for training and prediction.
    - Supports loading training data, training the model, and predicting test data.
    - Saves prediction results in a CSV file as required by the competition.

## Usage Instructions

### 1. Environment Setup

- Ensure Python 3.8 or higher is installed.
- Install required Python libraries:
  ```bash
  pip install -r requirements.txt
  ```

### 2. Dataset Preparation

- Datasets are provided in two formats: `csv` and `feather`.
  - **CSV Format**: Approximately 130G when decompressed, suitable for scenarios requiring direct access to or processing of raw text data.
  - **Feather Format**: Approximately 40G when decompressed, suitable for efficient data processing with better performance than CSV.
- Choose the format based on your needs and ensure it is decompressed to the correct path.

### 3. Configuration

- Configure data paths and other parameters in the `Config` class.
- Set `DATA_SUFFIX` based on the dataset format used:
  - For `csv` files, set `DATA_SUFFIX` to `csv`.
  - For `feather` files, set `DATA_SUFFIX` to `feather`.

### 4. Running the Baseline

- Execute the `baseline.py` script to:
  1. Initialize configuration.
  2. Extract and save features.
  3. Generate positive samples, negative samples, and test data.
  4. Train the model and perform predictions.
  5. Save the prediction results to a `submission.csv` file.

### 5. Using the Starter Kit

- Open the starter notebook:
  ```bash
  jupyter notebook starterkit_notebook.ipynb
  ```
- Follow the cells sequentially to:
  - Load and preprocess the dataset.
  - Perform feature engineering.
  - Train and evaluate a machine learning model.

### 6. Output Files

- **Feature Files**: Saved in the path specified by `feature_path`, in `.feather` format.
- **Training Data**: Positive and negative samples are saved in the path specified by `train_data_path`, in `.feather` format.
- **Test Data**: Saved in the path specified by `test_data_path`, in `.feather` format.
- **Prediction Results**: Saved in a `submission.csv` file, containing SN names, prediction timestamps, and SN types.

### 7. Submission Instructions

Compress the generated `submission.csv` file into a **zip** file and submit it to the [competition page](https://www.codabench.org/competitions/3586/).

## Notes

- Extend the starter kit or baseline as needed for better performance.
- Update paths and configurations in the `Config` class based on your local environment.

## License

This project is provided under the MIT License. See the LICENSE file for details.
