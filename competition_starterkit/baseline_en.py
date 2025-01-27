# -*- coding: utf-8 -*-
import abc
import os
from dataclasses import dataclass, field
from datetime import datetime
from multiprocessing import Pool
from typing import Dict, Tuple, NoReturn, Union, List

import feather
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm

# Define time constants (unit: seconds)
ONE_MINUTE = 60  # Number of seconds in one minute
ONE_HOUR = 3600  # Number of seconds in one hour (60 seconds * 60 minutes)
ONE_DAY = 86400  # Number of seconds in one day (60 seconds * 60 minutes * 24 hours)


@dataclass
class Config(object):
    """
    Configuration class for storing and managing program settings.
    Includes time window sizes, path settings, date ranges, feature extraction intervals, etc.
    """

    # Important! Set DATA_SUFFIX to 'csv' if using CSV files, or to 'feather' if using Feather files
    DATA_SUFFIX: str = field(default="feather", init=False)

    # Mapping of time window sizes, where keys are time lengths (in seconds) and values are their string representations
    TIME_WINDOW_SIZE_MAP: dict = field(
        default_factory=lambda: {
            15 * ONE_MINUTE: "15m",
            1 * ONE_HOUR: "1h"
        },
        init=False,
    )

    # List of time-related intervals, storing commonly used time intervals (in seconds)
    TIME_RELATED_LIST: List[int] = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR],
        init=False,
    )

    # Default value for imputing missing data
    IMPUTE_VALUE: int = field(default=-1, init=False)

    # Whether to use multiprocessing
    USE_MULTI_PROCESS: bool = field(default=True, init=False)

    # If using multiprocessing, the number of workers for parallel processing
    WORKER_NUM: int = field(default=16, init=False)

    # Path configurations: raw dataset path, generated feature path, processed training set feature path, processed test set feature path, and ticket path
    data_path: str = "To be filled"
    feature_path: str = "To be filled"
    train_data_path: str = "To be filled"
    test_data_path: str = "To be filled"
    ticket_path: str = "To be filled"

    # Date range configurations
    train_date_range: tuple = ("2024-01-01", "2024-06-01")
    test_data_range: tuple = ("2024-06-01", "2024-08-01")

    # The time interval (in seconds) for feature extraction. For better performance, it can be modified to 15 * ONE_MINUTE or 30 * ONE_MINUTE.
    feature_interval: int = ONE_HOUR


class FeatureFactory(object):
    """
    Feature factory class, used to generate features.
    """

    # Considering DDR4 memory, its DQ_COUNT and BURST_COUNT are 4 and 8, respectively.
    DQ_COUNT = 4
    BURST_COUNT = 8

    def __init__(self, config: Config):
        """
        Initialize the feature factory.

        :param config: Configuration class instance, containing paths and other information.
        """

        self.config = config
        os.makedirs(self.config.feature_path, exist_ok=True)
        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    def _unique_num_filtered(self, input_array: np.ndarray) -> int:
        """
        Deduplicate the input array, remove elements with the value IMPUTE_VALUE, and count the remaining unique elements.

        :param input_array: Input array
        :return: Number of unique elements after filtering
        """

        unique_array = np.unique(input_array)
        return len(unique_array) - int(self.config.IMPUTE_VALUE in unique_array)

    @staticmethod
    def _calculate_ce_storm_count(
        log_times: np.ndarray,
        ce_storm_interval_seconds: int = 60,
        ce_storm_count_threshold: int = 10,
    ) -> int:
        """
        Calculate the number of CE storms.

        CE storm definition:
        - Adjacent CE logs: If the time interval between two CE logs' LogTime is < 60s, they are considered adjacent logs.
        - If the number of adjacent logs exceeds 10, it is counted as one CE storm (note: if the number of adjacent logs continues to grow beyond 10, it is still counted as one CE storm).

        :param log_times: List of log LogTimes
        :param ce_storm_interval_seconds: Time interval threshold for CE storms
        :param ce_storm_count_threshold: Count threshold for CE storms
        :return: Number of CE storms
        """

        log_times = sorted(log_times)
        ce_storm_count = 0
        consecutive_count = 0

        for i in range(1, len(log_times)):
            if log_times[i] - log_times[i - 1] <= ce_storm_interval_seconds:
                consecutive_count += 1
            else:
                consecutive_count = 0
            if consecutive_count > ce_storm_count_threshold:
                ce_storm_count += 1
                consecutive_count = 0

        return ce_storm_count

    def _get_temporal_features(
        self, window_df: pd.DataFrame, time_window_size: int
    ) -> Dict[str, int]:
        """
        Extract temporal features, including CE count, log count, CE storm count, log occurrence frequency, etc.

        :param window_df: Data within the time window
        :param time_window_size: Size of the time window
        :return: Dictionary of temporal features

        - read_ce_log_num, read_ce_count: Total log count and CE count for read CEs within the time window
        - scrub_ce_log_num, scrub_ce_count: Total log count and CE count for scrub CEs within the time window
        - all_ce_log_num, all_ce_count: Total log count and CE count for all CEs within the time window
        - log_happen_frequency: Log occurrence frequency
        - ce_storm_count: Number of CE storms
        """

        error_type_is_READ_CE = window_df["error_type_is_READ_CE"].values
        error_type_is_SCRUB_CE = window_df["error_type_is_SCRUB_CE"].values
        ce_count = window_df["Count"].values

        temporal_features = dict()
        temporal_features["read_ce_log_num"] = error_type_is_READ_CE.sum()
        temporal_features["scrub_ce_log_num"] = error_type_is_SCRUB_CE.sum()
        temporal_features["all_ce_log_num"] = len(window_df)

        temporal_features["read_ce_count"] = (error_type_is_READ_CE * ce_count).sum()
        temporal_features["scrub_ce_count"] = (error_type_is_SCRUB_CE * ce_count).sum()
        temporal_features["all_ce_count"] = ce_count.sum()

        temporal_features["log_happen_frequency"] = (
            time_window_size / len(window_df) if not window_df.empty else 0
        )
        temporal_features["ce_storm_count"] = self._calculate_ce_storm_count(
            window_df["LogTime"].values
        )
        return temporal_features

    def _get_spatio_features(self, window_df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract spatial features, including fault modes and the number of simultaneous row/column faults.

        :param window_df: Data within the time window
        :return: Dictionary of spatial features

        - fault_mode_others: Other faults, where multiple devices have faults
        - fault_mode_device: Device faults, where multiple banks in the same device have faults
        - fault_mode_bank: Bank faults, where multiple rows in the same bank have faults
        - fault_mode_row: Row faults, where multiple cells in the same row have faults
        - fault_mode_column: Column faults, where multiple cells in the same column have faults
        - fault_mode_cell: Cell faults, where multiple cells with the same ID have faults
        - fault_row_num: Number of rows with simultaneous row faults
        - fault_column_num: Number of columns with simultaneous column faults
        """

        spatio_features = {
            "fault_mode_others": 0,
            "fault_mode_device": 0,
            "fault_mode_bank": 0,
            "fault_mode_row": 0,
            "fault_mode_column": 0,
            "fault_mode_cell": 0,
            "fault_row_num": 0,
            "fault_column_num": 0,
        }

        # Determine fault mode based on the number of faulty devices, banks, rows, columns, and cells
        if self._unique_num_filtered(window_df["deviceID"].values) > 1:
            spatio_features["fault_mode_others"] = 1
        elif self._unique_num_filtered(window_df["BankId"].values) > 1:
            spatio_features["fault_mode_device"] = 1
        elif (
            self._unique_num_filtered(window_df["ColumnId"].values) > 1
            and self._unique_num_filtered(window_df["RowId"].values) > 1
        ):
            spatio_features["fault_mode_bank"] = 1
        elif self._unique_num_filtered(window_df["ColumnId"].values) > 1:
            spatio_features["fault_mode_row"] = 1
        elif self._unique_num_filtered(window_df["RowId"].values) > 1:
            spatio_features["fault_mode_column"] = 1
        elif self._unique_num_filtered(window_df["CellId"].values) == 1:
            spatio_features["fault_mode_cell"] = 1

        # Record column address information for the same row
        row_pos_dict = {}
        # Record row address information for the same column
        col_pos_dict = {}

        for device_id, bank_id, row_id, column_id in zip(
            window_df["deviceID"].values,
            window_df["BankId"].values,
            window_df["RowId"].values,
            window_df["ColumnId"].values,
        ):
            current_row = "_".join([str(pos) for pos in [device_id, bank_id, row_id]])
            current_col = "_".join(
                [str(pos) for pos in [device_id, bank_id, column_id]]
            )
            row_pos_dict.setdefault(current_row, [])
            col_pos_dict.setdefault(current_col, [])
            row_pos_dict[current_row].append(column_id)
            col_pos_dict[current_col].append(row_id)

        for row in row_pos_dict:
            if self._unique_num_filtered(np.array(row_pos_dict[row])) > 1:
                spatio_features["fault_row_num"] += 1
        for col in col_pos_dict:
            if self._unique_num_filtered(np.array(col_pos_dict[col])) > 1:
                spatio_features["fault_column_num"] += 1

        return spatio_features

    @staticmethod
    def _get_err_parity_features(window_df: pd.DataFrame) -> Dict[str, int]:
        """
        Extract parity-related features.

        :param window_df: Data within the time window.
        :return: Dictionary of parity features

        - error_bit_count: Total number of error bits within the time window.
        - error_dq_count: Total number of DQ errors within the time window.
        - error_burst_count: Total number of burst errors within the time window.
        - max_dq_interval: Maximum DQ error interval for each parity within the time window.
        - max_burst_interval: Maximum burst error interval for each parity within the time window.
        - dq_count=n: Total count of DQ errors equal to n, where n ranges from [1, 2, 3, 4]. Default: 0.
        - burst_count=n: Total count of burst errors equal to n, where n ranges from [1, 2, 3, 4, 5, 6, 7, 8]. Default: 0.
        """

        err_parity_features = dict()

        # Calculate total error bit count, DQ error count, and burst error count
        err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum()
        err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
        err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum()

        # Calculate maximum DQ interval and maximum burst interval
        err_parity_features["max_dq_interval"] = window_df[
            "max_dq_interval"
        ].values.max()
        err_parity_features["max_burst_interval"] = window_df[
            "max_burst_interval"
        ].values.max()

        # Count the distribution of DQ errors and burst errors
        dq_counts = dict()
        burst_counts = dict()
        for dq, burst in zip(
            window_df["dq_count"].values, window_df["burst_count"].values
        ):
            dq_counts[dq] = dq_counts.get(dq, 0) + 1
            burst_counts[burst] = burst_counts.get(burst, 0) + 1

        # Calculate the total count of DQ errors equal to n, where n ranges from [1, 2, 3, 4]
        for dq in range(1, FeatureFactory.DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)

        # Calculate the total count of burst errors equal to n, where n ranges from [1, 2, 3, 4, 5, 6, 7, 8]
        for burst in range(1, FeatureFactory.BURST_COUNT + 1):
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

        return err_parity_features

    @staticmethod
    def _get_bit_dq_burst_info(parity: np.int64) -> Tuple[int, int, int, int, int]:
        """
        Extract parity-related information for a specific parity value.

        :param parity: The parity value
        :return: A tuple containing the following parity-related information

        - bit_count: Number of error bits in the parity.
        - dq_count: Number of DQ errors in the parity.
        - burst_count: Number of burst errors in the parity.
        - max_dq_interval: Maximum DQ error interval in the parity.
        - max_burst_interval: Maximum burst error interval in the parity.
        """

        # Convert parity to a 32-bit binary string
        bin_parity = bin(parity)[2:].zfill(32)

        # Calculate the number of error bits
        bit_count = bin_parity.count("1")

        # Calculate burst-related features
        binary_row_array = [bin_parity[i : i + 4].count("1") for i in range(0, 32, 4)]
        binary_row_array_indices = [
            idx for idx, value in enumerate(binary_row_array) if value > 0
        ]
        burst_count = len(binary_row_array_indices)
        max_burst_interval = (
            binary_row_array_indices[-1] - binary_row_array_indices[0]
            if binary_row_array_indices
            else 0
        )

        # Calculate DQ-related features
        binary_column_array = [bin_parity[i::4].count("1") for i in range(4)]
        binary_column_array_indices = [
            idx for idx, value in enumerate(binary_column_array) if value > 0
        ]
        dq_count = len(binary_column_array_indices)
        max_dq_interval = (
            binary_column_array_indices[-1] - binary_column_array_indices[0]
            if binary_column_array_indices
            else 0
        )

        return bit_count, dq_count, burst_count, max_dq_interval, max_burst_interval

    def _get_processed_df(self, sn_file: str) -> pd.DataFrame:
        """
        Generate a processed DataFrame.

        Processing steps include:
        - Sorting raw_df by LogTime.
        - Converting error_type to one-hot encoding.
        - Filling missing values.
        - Adding parity-related features.

        :param sn_file: SN file name.
        :return: Processed DataFrame.
        """

        parity_dict = dict()

        # Read raw data and sort by LogTime
        if self.config.DATA_SUFFIX == "csv":
            raw_df = pd.read_csv(os.path.join(self.config.data_path, sn_file))
        else:
            raw_df = feather.read_dataframe(os.path.join(self.config.data_path, sn_file))

        raw_df = raw_df.sort_values(by="LogTime").reset_index(drop=True)

        # Extract necessary columns and initialize processed_df
        processed_df = raw_df[
            [
                "LogTime",
                "deviceID",
                "BankId",
                "RowId",
                "ColumnId",
                "MciAddr",
                "RetryRdErrLogParity",
            ]
        ].copy()

        # Fill missing values in deviceID and convert to integer
        processed_df["deviceID"] = (
            processed_df["deviceID"].fillna(self.config.IMPUTE_VALUE).astype(int)
        )

        # Convert error_type to one-hot encoding
        processed_df["error_type_is_READ_CE"] = (
            raw_df["error_type_full_name"] == "CE.READ"
        ).astype(int)
        processed_df["error_type_is_SCRUB_CE"] = (
            raw_df["error_type_full_name"] == "CE.SCRUB"
        ).astype(int)

        processed_df["CellId"] = (
            processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
        )
        processed_df["position_and_parity"] = (
            processed_df["deviceID"].astype(str)
            + "_"
            + processed_df["BankId"].astype(str)
            + "_"
            + processed_df["RowId"].astype(str)
            + "_"
            + processed_df["ColumnId"].astype(str)
            + "_"
            + processed_df["RetryRdErrLogParity"].astype(str)
        )

        err_log_parity_array = (
            processed_df["RetryRdErrLogParity"]
            .fillna(0)
            .replace("", 0)
            .astype(np.int64)  # Convert to np.int64 to avoid overflow
            .values
        )

        # Calculate bit_count, dq_count, burst_count, max_dq_interval, and max_burst_interval for each parity
        bit_dq_burst_count = list()
        for idx, err_log_parity in enumerate(err_log_parity_array):
            if err_log_parity not in parity_dict:
                parity_dict[err_log_parity] = self._get_bit_dq_burst_info(
                    err_log_parity
                )
            bit_dq_burst_count.append(parity_dict[err_log_parity])

        processed_df = processed_df.join(
            pd.DataFrame(
                bit_dq_burst_count,
                columns=[
                    "bit_count",
                    "dq_count",
                    "burst_count",
                    "max_dq_interval",
                    "max_burst_interval",
                ],
            )
        )
        return processed_df

    def process_single_sn(self, sn_file: str) -> NoReturn:
        """
        Process a single SN file to extract features for different time window scales.

        :param sn_file: SN file name
        """

        # Get the processed DataFrame
        new_df = self._get_processed_df(sn_file)

        # Calculate the time index based on the feature generation interval
        new_df["time_index"] = new_df["LogTime"] // self.config.feature_interval
        log_times = new_df["LogTime"].values

        # Calculate the end and start times for each time window, using a maximum of max_window_size historical data
        max_window_size = max(self.config.TIME_RELATED_LIST)
        window_end_times = new_df.groupby("time_index")["LogTime"].max().values
        window_start_times = window_end_times - max_window_size

        # Find the corresponding data indices based on the start and end times of the time window
        start_indices = np.searchsorted(log_times, window_start_times, side="left")
        end_indices = np.searchsorted(log_times, window_end_times, side="right")

        combined_dict_list = []
        for start_idx, end_idx, end_time in zip(
            start_indices, end_indices, window_end_times
        ):
            combined_dict = {}
            window_df = new_df.iloc[start_idx:end_idx]
            combined_dict["LogTime"] = window_df["LogTime"].values.max()

            # Calculate the occurrence count for each position_and_parity and remove duplicates
            window_df = window_df.assign(
                Count=window_df.groupby("position_and_parity")[
                    "position_and_parity"
                ].transform("count")
            )
            window_df = window_df.drop_duplicates(
                subset="position_and_parity", keep="first"
            )
            log_times = window_df["LogTime"].values
            end_logtime_of_filtered_window_df = window_df["LogTime"].values.max()

            # Iterate over different time window sizes to extract features (corresponding to max_window_size, ensuring the time window does not exceed max_window_size)
            for time_window_size in self.config.TIME_RELATED_LIST:
                index = np.searchsorted(
                    log_times,
                    end_logtime_of_filtered_window_df - time_window_size,
                    side="left",
                )
                window_df_copy = window_df.iloc[index:]

                # Extract temporal, spatial, and parity-related features
                temporal_features = self._get_temporal_features(
                    window_df_copy, time_window_size
                )
                spatio_features = self._get_spatio_features(window_df_copy)
                err_parity_features = self._get_err_parity_features(window_df_copy)

                # Merge features into combined_dict and append the time window size as a suffix
                combined_dict.update(
                    {
                        f"{key}_{self.config.TIME_WINDOW_SIZE_MAP[time_window_size]}": value
                        for d in [
                            temporal_features,
                            spatio_features,
                            err_parity_features,
                        ]
                        for key, value in d.items()
                    }
                )
            combined_dict_list.append(combined_dict)

        # Convert the feature list to a DataFrame and save it as a feather file
        combined_df = pd.DataFrame(combined_dict_list)
        feather.write_dataframe(
            combined_df,
            os.path.join(self.config.feature_path, sn_file.replace("csv", "feather")),
        )

    def process_all_sn(self) -> NoReturn:
        """
        Process all SN files and save the features. Supports multiprocessing to improve efficiency.
        """

        sn_files = os.listdir(self.config.data_path)
        exist_sn_file_list = os.listdir(self.config.feature_path)
        sn_files = [
            x for x in sn_files if x not in exist_sn_file_list and x.endswith(self.config.DATA_SUFFIX)
        ]
        sn_files.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                list(
                    tqdm(
                        pool.imap(self.process_single_sn, sn_files),
                        total=len(sn_files),
                        desc="Generating features",
                    )
                )
        else:
            for sn_file in tqdm(sn_files, desc="Generating features"):
                self.process_single_sn(sn_file)


class DataGenerator(metaclass=abc.ABCMeta):
    """
    Base class for data generators, used to generate training and testing data.
    """

    # Chunk size for processing data in batches
    CHUNK_SIZE = 200

    def __init__(self, config: Config):
        """
        Initialize the data generator.

        :param config: Configuration class instance, containing paths, date ranges, etc.
        """

        self.config = config
        self.feature_path = self.config.feature_path
        self.train_data_path = self.config.train_data_path
        self.test_data_path = self.config.test_data_path
        self.ticket_path = self.config.ticket_path

        # Convert date ranges to timestamps
        self.train_start_date = self._datetime_to_timestamp(
            self.config.train_date_range[0]
        )
        self.train_end_date = self._datetime_to_timestamp(
            self.config.train_date_range[1]
        )
        self.test_start_date = self._datetime_to_timestamp(
            self.config.test_data_range[0]
        )
        self.test_end_date = self._datetime_to_timestamp(self.config.test_data_range[1])

        ticket = pd.read_csv(self.ticket_path)
        ticket = ticket[ticket["alarm_time"] <= self.train_end_date]
        self.ticket = ticket
        self.ticket_sn_map = {
            sn: sn_t
            for sn, sn_t in zip(list(ticket["sn_name"]), list(ticket["alarm_time"]))
        }

        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    @staticmethod
    def concat_in_chunks(chunks: List) -> Union[pd.DataFrame, None]:
        """
        Concatenate DataFrames in chunks.

        :param chunks: List of DataFrames
        :return: Concatenated DataFrame, or None if chunks is empty
        """

        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            return pd.concat(chunks)
        return None

    def parallel_concat(
        self, results: List, chunk_size: int = CHUNK_SIZE
    ) -> Union[pd.DataFrame, None]:
        """
        Parallelized concatenation operation, can be seen as a parallelized version of concat_in_chunks.

        :param results: List of results to concatenate
        :param chunk_size: Size of each chunk
        :return: Concatenated DataFrame
        """

        chunks = [
            results[i : i + chunk_size] for i in range(0, len(results), chunk_size)
        ]

        # Use multiprocessing for parallel concatenation
        worker_num = self.config.WORKER_NUM
        with Pool(worker_num) as pool:
            concatenated_chunks = pool.map(self.concat_in_chunks, chunks)

        return self.concat_in_chunks(concatenated_chunks)

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        """
        Convert a date string in %Y-%m-%d format to a timestamp.

        :param date: Date string
        :return: Timestamp
        """

        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def _get_data(self) -> pd.DataFrame:
        """
        Retrieve all data under feature_path and process it.

        :return: Processed data
        """

        file_list = os.listdir(self.feature_path)
        file_list = [x for x in file_list if x.endswith(".feather")]
        file_list.sort()

        if self.config.USE_MULTI_PROCESS:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                results = list(
                    tqdm(
                        pool.imap(self._process_file, file_list),
                        total=len(file_list),
                        desc="Processing files",
                    )
                )
            data_all = self.parallel_concat(results)
        else:
            data_all = []
            data_chunk = []
            for i in tqdm(range(len(file_list)), desc="Processing files"):
                data = self._process_file(file_list[i])
                if data is not None:
                    data_chunk.append(data)
                if len(data_chunk) >= self.CHUNK_SIZE:
                    data_all.append(self.concat_in_chunks(data_chunk))
                    data_chunk = []
            if data_chunk:
                data_all.append(pd.concat(data_chunk))
            data_all = pd.concat(data_all)

        return data_all

    @abc.abstractmethod
    def _process_file(self, sn_file):
        """
        Process a single file. Subclasses must implement this method.

        :param sn_file: File name
        """

        raise NotImplementedError("Subclasses should implement this method")

    @abc.abstractmethod
    def generate_and_save_data(self):
        """
        Generate and save data. Subclasses must implement this method.
        """

        raise NotImplementedError("Subclasses should implement this method")


class PositiveDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        Process a single file to extract positive sample data.

        :param sn_file: File name
        :return: Processed DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if self.ticket_sn_map.get(sn_name):
            # Set the time range for positive samples to 30 days before the ticket time
            end_time = self.ticket_sn_map.get(sn_name)
            start_time = end_time - 30 * ONE_DAY

            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            data["label"] = 1

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # If SN name is not in the ticket, return None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        Generate and save positive sample data.
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "positive_train.feather")
        )


class NegativeDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        Process a single file to extract negative sample data.

        :param sn_file: File name
        :return: Processed DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]
        if not self.ticket_sn_map.get(sn_name):
            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))

            # Set the time range for negative samples to a continuous 30-day period
            end_time = self.train_end_date - 30 * ONE_DAY
            start_time = self.train_end_date - 60 * ONE_DAY

            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            if data.empty:
                return None
            data["label"] = 0

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data

        # If SN name is in the ticket, return None
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        Generate and save negative sample data.
        """

        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "negative_train.feather")
        )


class TestDataGenerator(DataGenerator):
    @staticmethod
    def _split_dataframe(df: pd.DataFrame, chunk_size: int = 2000000):
        """
        Split a DataFrame into chunks of specified size.

        :param df: DataFrame to be split
        :param chunk_size: Size of each chunk
        :return: Split DataFrame, yielding one chunk at a time
        """

        for start in range(0, len(df), chunk_size):
            yield df[start : start + chunk_size]

    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        Process a single file to extract test data.

        :param sn_file: File name
        :return: Processed DataFrame
        """

        sn_name = os.path.splitext(sn_file)[0]

        # Read the feature file and filter data within the test time range
        data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
        data = data[data["LogTime"] >= self.test_start_date]
        data = data[data["LogTime"] <= self.test_end_date]
        if data.empty:
            return None

        index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
        data.index = pd.MultiIndex.from_tuples(index_list)
        return data

    def generate_and_save_data(self) -> NoReturn:
        """
        Generate and save test data.
        """

        data_all = self._get_data()
        for index, chunk in enumerate(self._split_dataframe(data_all)):
            feather.write_dataframe(
                chunk, os.path.join(self.test_data_path, f"res_{index}.feather")
            )


class MFPmodel(object):
    """
    Memory Failure Prediction Model Class
    """

    def __init__(self, config: Config):
        """
        Initialize the model class.

        :param config: An instance of the Config class, containing paths to training and testing data, etc.
        """

        self.train_data_path = config.train_data_path
        self.test_data_path = config.test_data_path
        self.model_params = {
            "learning_rate": 0.02,
            "n_estimators": 500,
            "max_depth": 8,
            "num_leaves": 20,
            "min_child_samples": 20,
            "verbose": 1,
        }
        self.model = LGBMClassifier(**self.model_params)

    def load_train_data(self) -> NoReturn:
        """
        Load training data.
        """

        self.train_pos = feather.read_dataframe(
            os.path.join(self.train_data_path, "positive_train.feather")
        )
        self.train_neg = feather.read_dataframe(
            os.path.join(self.train_data_path, "negative_train.feather")
        )

    def train(self) -> NoReturn:
        """
        Train the model.
        """

        train_all = pd.concat([self.train_pos, self.train_neg])
        train_all.drop("LogTime", axis=1, inplace=True)
        train_all = train_all.sort_index(axis=1)

        self.model.fit(train_all.drop(columns=["label"]), train_all["label"])

    def predict_proba(self) -> Dict[str, List]:
        """
        Predict the probability of each test sample being classified as positive, and return the result.

        :return: A dictionary where the key is `sn_name` and the value is a list of predicted probabilities for being positive.
        """
        result = {}
        for file in os.listdir(self.test_data_path):
            test_df = feather.read_dataframe(os.path.join(self.test_data_path, file))
            test_df["sn_name"] = test_df.index.get_level_values(0)
            test_df["log_time"] = test_df.index.get_level_values(1)

            test_df = test_df[self.model.feature_name_]
            predict_result = self.model.predict_proba(test_df)

            index_list = list(test_df.index)
            for i in tqdm(range(len(index_list))):
                p_s = predict_result[i][1]

                # Filter low-probability samples to reduce memory usage of predictions
                if p_s < 0.1:
                    continue

                sn = index_list[i][0]
                sn_t = datetime.fromtimestamp(index_list[i][1])
                result.setdefault(sn, [])
                result[sn].append((sn_t, p_s))
        return result

    def predict(self, threshold: int = 0.5) -> Dict[str, List]:
        """
        Get prediction results based on a specific threshold.

        :param threshold: The threshold value, default is 0.5.
        :return: A dictionary where the key is `sn_name` and the value is a list of timestamps filtered by the threshold.
        """

        # Get the probability prediction results
        result = self.predict_proba()

        # Filter the prediction results based on the threshold
        result = {
            sn: [int(sn_t.timestamp()) for sn_t, p_s in pred_list if p_s >= threshold]
            for sn, pred_list in result.items()
        }

        # Filter out empty prediction results and sort the results by time
        result = {
            sn: sorted(pred_list) for sn, pred_list in result.items() if pred_list
        }

        return result


if __name__ == "__main__":
    sn_type = "A"  # SN type, A or B, here using type A as an example
    test_stage = 1  # Test stage, 1 or 2, here using Stage 1 as an example

    # Set the time range for test data based on the test stage
    if test_stage == 1:
        test_data_range: tuple = ("2024-06-01", "2024-08-01")  # Time range for Stage 1 test data
    else:
        test_data_range: tuple = ("2024-08-01", "2024-10-01")  # Time range for Stage 2 test data

    # Initialize the Config class, setting data paths, feature paths, training data paths, test data paths, etc.
    config = Config(
        data_path=os.path.join("D:/competition_data/stage1_feather", f"type_{sn_type}"),  # Path to the raw dataset
        feature_path=os.path.join(
            "D:/release_features/combined_sn_feature", f"type_{sn_type}"  # Path to the generated feature data
        ),
        train_data_path=os.path.join(
            "D:/release_features/train_data", f"type_{sn_type}"  # Path to the generated training data
        ),
        test_data_path=os.path.join(
            "D:/release_features/test_data", f"type_{sn_type}_{test_stage}"  # Path to the generated test data
        ),
        test_data_range=test_data_range,  # Time range for test data
        ticket_path="D:/competition_data/stage1_feather/ticket.csv",  # Path to the ticket data
    )

    # Initialize the FeatureFactory class to process SN files and generate features
    feature_factory = FeatureFactory(config)
    feature_factory.process_all_sn()  # Process all SN files

    # Initialize the positive data generator to generate and save positive data
    positive_data_generator = PositiveDataGenerator(config)
    positive_data_generator.generate_and_save_data()

    # Initialize the negative data generator to generate and save negative data
    negative_data_generator = NegativeDataGenerator(config)
    negative_data_generator.generate_and_save_data()

    # Initialize the test data generator to generate and save test data
    test_data_generator = TestDataGenerator(config)
    test_data_generator.generate_and_save_data()

    # Initialize the MFPmodel class, load training data, and train the model
    model = MFPmodel(config)
    model.load_train_data()  # Load training data
    model.train()  # Train the model
    result = model.predict()  # Use the trained model to make predictions

    # Convert prediction results into submission format
    submission = []
    for sn in result:  # Iterate through prediction results for each SN
        for timestamp in result[sn]:  # Iterate through each timestamp
            submission.append([sn, timestamp, sn_type])  # Add SN name, prediction timestamp, and SN type

    # Convert submission data into a DataFrame and save it as a CSV file
    submission = pd.DataFrame(
        submission, columns=["sn_name", "prediction_timestamp", "serial_number_type"]
    )
    submission.to_csv("submission.csv", index=False, encoding="utf-8")

    print()
