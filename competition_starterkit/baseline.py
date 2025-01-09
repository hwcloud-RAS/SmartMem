# encoding: utf-8
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

# 定义常量, 用于表示时, 分, 秒的换算关系
ONE_MINUTE = 60
ONE_HOUR = 3600
ONE_DAY = 86400


@dataclass
class Config(object):
    """
    配置类, 用于表示配置信息
    """

    TIME_WINDOW_SIZE_MAP: dict = field(
        default_factory=lambda: {
            15 * ONE_MINUTE: "15m",
            ONE_HOUR: "1h",
            6 * ONE_HOUR: "6h",
        },
        init=False,
    )
    TIME_RELATED_LIST: list = field(
        default_factory=lambda: [15 * ONE_MINUTE, ONE_HOUR, 6 * ONE_HOUR], init=False
    )
    IMPUTE_VALUE: int = field(default=-1, init=False)
    WORKER_NUM: int = field(default=16, init=False)

    data_path: str = "D:/competition_data/type_A"
    feature_path: str = "D:/release_features/combined_sn_feature"
    train_data_path: str = "D:/release_features/train_data/type_A"
    test_data_path: str = "D:/release_features/test_data/type_A"
    ticket_path: str = "D:/competition_data/ticket.csv"
    train_date_range: tuple = ("2024-01-01", "2024-06-01")
    test_data_range: tuple = ("2024-06-01", "2024-08-01")

    feature_interval: int = 15 * ONE_MINUTE
    use_multi_process: bool = True


class FeatureFactory(object):
    """
    特征工厂类, 用于生成特征
    """

    DQ_COUNT = 4
    BURST_COUNT = 8

    def __init__(self, config: Config):
        self.config = config
        os.makedirs(self.config.feature_path, exist_ok=True)
        os.makedirs(self.config.train_data_path, exist_ok=True)
        os.makedirs(self.config.test_data_path, exist_ok=True)

    def _unique_num_filtered(self, input_array: np.ndarray) -> int:
        """
        对输入的列表进行过滤, 去除值为 IMPUTE_VALUE 的元素后, 统计元素个数

        :param input_array: 输入的列表
        :return: 返回经过过滤后的列表元素个数
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
        计算 CE 风暴的数量, CE 风暴定义:
        - 首先定义相邻 CE 日志: 若两个 CE 日志 LogTime 时间间隔 < 60s, 则为相邻日志;
        - 如果相邻日志的个数 >10, 则为发生 1 次 CE 风暴(注意: 如果相邻日志数量持续增长, 超过了 10, 则也只是记作 1 次 CE 风暴)

        :param log_times: 日志 LogTime 列表
        :param ce_storm_interval_seconds: CE 风暴的时间间隔阈值
        :param ce_storm_count_threshold: CE 风暴的数量阈值
        :return: CE风暴的数量
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
        获取时间特征, 包括 CE 数量, 日志数量, CE 风暴数量, 日志发生频率等

        :param window_df: 时间窗口内的数据
        :param time_window_size: 时间窗口大小
        :return: 时间特征

        - read_ce_log_num, read_ce_count: 时间窗口内, 读 CE 的 count 总数, 日志总数
        - scrub_ce_log_num, scrub_ce_count: 时间窗口内, 巡检 CE 的 count 总数, 日志总数
        - all_ce_log_num, all_ce_count: 时间窗口内, 所有 CE 的 count 总数, 日志总数
        - log_happen_frequency: 日志发生频率
        - ce_storm_count: CE 风暴数量
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
        获取空间特征, 包括故障模式, 同时发生行列故障的数量
        :param window_df: 时间窗口内的数据
        :return: 空间特征
        fault_mode_others: 其他故障, 即多个 device 发生故障
        fault_mode_device: device 故障, 相同 id 的 device 发生多个 bank 故障
        fault_mode_bank: bank 故障, 相同 id 的 bank 发生多个 row故障
        fault_mode_row: row 故障, 相同 id 的 row 发生多个不同 column 的 cell 故障
        fault_mode_column: column 故障, 相同 id 的 column 发生多个不同 row 的 cell 故障
        fault_mode_cell: cell 故障, 发生多个相同 id 的 cell 故障
        fault_row_num: 同时发生 row 故障的行个数
        fault_column_num: 同时发生 column 故障的列个数
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

        # 记录相同行对应的列地址信息
        row_pos_dict = {}
        # 记录相同列对应的行地址信息
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
        获取奇偶校验特征
        :param window_df: 时间窗口内的数据
        :return: 奇偶校验特征
        error_bit_count: 时间窗口内, 总错误 bit 数
        error_dq_count: 时间窗口内, 总 dq 错误数
        error_burst_count: 时间窗口内, 总 burst 错误数
        max_dq_interval: 时间窗口内, 每个 parity 最大错误 dq 距离的最大值
        max_burst_interval: 时间窗口内, 每个 parity 最大错误 burst 距离的最大值
        dq_count=n: dq错误数=n 的总数量, n 取值 [1,2,3,4],默认值: 0
        burst_count=n: burst错误数=n 的总数量, n 取值 [1,2,3,4,5,6,7,8],默认值: 0
        """
        err_parity_features = dict()

        err_parity_features["error_bit_count"] = window_df["bit_count"].values.sum()
        err_parity_features["error_dq_count"] = window_df["dq_count"].values.sum()
        err_parity_features["error_burst_count"] = window_df["burst_count"].values.sum()
        err_parity_features["max_dq_interval"] = window_df[
            "max_dq_interval"
        ].values.max()
        err_parity_features["max_burst_interval"] = window_df[
            "max_burst_interval"
        ].values.max()

        dq_counts = dict()
        burst_counts = dict()
        for i in zip(window_df["dq_count"].values, window_df["burst_count"].values):
            dq_counts[i[0]] = dq_counts.get(i[0], 0) + 1
            burst_counts[i[1]] = burst_counts.get(i[1], 0) + 1

        # 计算 'dq错误数=n' 的总数量, DDR4 内存的 DQ_COUNT 为 4, 因此 n 取值 [1,2,3,4]
        for dq in range(1, FeatureFactory.DQ_COUNT + 1):
            err_parity_features[f"dq_count={dq}"] = dq_counts.get(dq, 0)
        # 计算 'burst错误数=n' 的总数量, DDR4 内存的 BURST_COUNT 为 8, 因此 n 取值 [1,2,3,4,5,6,7,8]
        for burst in [1, 2, 3, 4, 5, 6, 7, 8]:
            err_parity_features[f"burst_count={burst}"] = burst_counts.get(burst, 0)

        return err_parity_features

    @staticmethod
    def _get_bit_dq_burst_info(parity: np.int64) -> Tuple[int, int, int, int, int]:
        """
        获取特定 parity 的奇偶校验信息
        :param parity: 奇偶校验值
        :return: parity 的奇偶校验信息
        bit_count: parity 错误 bit 数量
        dq_count: parity 错误 dq 数量
        burst_count: parity 错误 burst 数量
        max_dq_interval: parity 错误 dq 的最大间隔
        max_burst_interval: parity 错误 burst 的最大间隔
        """
        bin_parity = bin(parity)[2:].zfill(32)
        bit_count = bin_parity.count("1")
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
        获取处理后的 DataFrame, 处理包括对 raw_df 排序, error_type 转独热编码, 填充缺失值, 添加奇偶校验特征
        :param sn_file: sn 文件名
        :return: 处理后的 DataFrame
        """
        parity_dict = dict()
        raw_df = pd.read_csv(os.path.join(self.config.data_path, sn_file))
        raw_df = raw_df.sort_values(by="LogTime").reset_index(drop=True)
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
        processed_df["deviceID"] = (
            processed_df["deviceID"].fillna(self.config.IMPUTE_VALUE).astype(int)
        )
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
            .astype(np.int64)
            .values
        )
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
        处理单个 sn 文件, 获取不同尺度的时间窗口特征
        :param sn_file: sn 文件名
        """
        new_df = self._get_processed_df(sn_file)
        new_df["time_index"] = new_df["LogTime"] // self.config.feature_interval
        log_times = new_df["LogTime"].values

        window_end_times = new_df.groupby("time_index")["LogTime"].max().values
        window_start_times = window_end_times - 6 * ONE_HOUR

        start_indices = np.searchsorted(log_times, window_start_times, side="left")
        end_indices = np.searchsorted(log_times, window_end_times, side="right")

        combined_dict_list = []
        for start_idx, end_idx, end_time in zip(
            start_indices, end_indices, window_end_times
        ):
            combined_dict = {}
            window_df = new_df.iloc[start_idx:end_idx]
            combined_dict["LogTime"] = window_df["LogTime"].values.max()
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

            for time_window_size in self.config.TIME_RELATED_LIST:
                index = np.searchsorted(
                    log_times,
                    end_logtime_of_filtered_window_df - time_window_size,
                    side="left",
                )
                window_df_copy = window_df.iloc[index:]

                temporal_features = self._get_temporal_features(
                    window_df_copy, time_window_size
                )
                spatio_features = self._get_spatio_features(window_df_copy)
                err_parity_features = self._get_err_parity_features(window_df_copy)

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
        combined_df = pd.DataFrame(combined_dict_list)
        feather.write_dataframe(
            combined_df, os.path.join(self.config.feature_path, sn_file)
        )

    def process_all_sn(self) -> NoReturn:
        """
        处理所有 sn 文件, 并保存特征
        """
        sn_files = os.listdir(self.config.data_path)
        exist_sn_file_list = os.listdir(self.config.feature_path)
        sn_files = [
            x for x in sn_files if x not in exist_sn_file_list and x.endswith(".csv")
        ]
        sn_files.sort()

        if self.config.use_multi_process:
            worker_num = self.config.WORKER_NUM
            with Pool(worker_num) as pool:
                list(
                    tqdm(
                        pool.imap(self.process_single_sn, sn_files),
                        total=len(sn_files),
                        desc="Processing files",
                    )
                )
        else:
            for sn_file in tqdm(sn_files, desc="Processing files"):
                self.process_single_sn(sn_file)


# 生成训练、测试数据类
class DataGenerator(metaclass=abc.ABCMeta):
    """
    数据生成器基类
    """

    CHUNK_SIZE = 200

    def __init__(self, config: Config):
        self.config = config

        self.feature_path = self.config.feature_path
        self.train_data_path = self.config.train_data_path
        self.test_data_path = self.config.test_data_path
        self.ticket_path = self.config.ticket_path

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
        将 chunks 中的 DataFrame 进行拼接
        :param chunks: DataFrame 列表
        :return: 拼接后的 DataFrame
        """
        chunks = [chunk for chunk in chunks if chunk is not None]
        if chunks:
            return pd.concat(chunks)
        return None

    def parallel_concat(
        self, results: List, chunk_size: int = CHUNK_SIZE
    ) -> Union[pd.DataFrame, None]:
        """
        并行化的拼接操作, 可以视为 concat_in_chunks 的并行化版本
        :param results: 需要拼接的结果
        :param chunk_size: 拼接的 chunk 大小
        :return: 拼接后的 DataFrame
        """
        chunks = [
            results[i : i + chunk_size] for i in range(0, len(results), chunk_size)
        ]

        worker_num = self.config.WORKER_NUM
        with Pool(worker_num) as pool:
            concatenated_chunks = pool.map(self.concat_in_chunks, chunks)

        return self.concat_in_chunks(concatenated_chunks)

    @staticmethod
    def _datetime_to_timestamp(date: str) -> int:
        """
        将 %Y-%m-%d 格式的日期转换为时间戳
        :param date: 日期
        :return: 时间戳
        """
        return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

    def _get_data(self) -> pd.DataFrame:
        """
        获取 feature_path 下的所有数据, 并进行处理
        :return: 处理后的数据
        """
        file_list = os.listdir(self.feature_path)
        file_list = [x for x in file_list if x.endswith(".csv")]
        file_list.sort()

        if self.config.use_multi_process:
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
        处理单个文件, 子类需要实现该方法
        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """
        raise NotImplementedError("Subclasses should implement this method")

    @abc.abstractmethod
    def generate_and_save_data(self):
        """
        生成并保存数据, 子类需要实现该方法
        """
        raise NotImplementedError("Subclasses should implement this method")


class PositiveDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取正样本数据
        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """
        sn_name = os.path.splitext(sn_file)[0]
        if self.ticket_sn_map.get(sn_name):
            end_time = self.ticket_sn_map.get(sn_name)
            start_time = end_time - 30 * ONE_DAY

            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            data["label"] = 1

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存正样本数据
        """
        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "positive_train.csv")
        )


class NegativeDataGenerator(DataGenerator):
    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取负样本数据
        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """
        sn_name = os.path.splitext(sn_file)[0]
        if not self.ticket_sn_map.get(sn_name):
            data = feather.read_dataframe(os.path.join(self.feature_path, sn_file))
            end_time = self.train_end_date - 30 * ONE_DAY
            start_time = self.train_end_date - 60 * ONE_DAY

            data = data[(data["LogTime"] <= end_time) & (data["LogTime"] >= start_time)]
            if data.empty:
                return None
            data["label"] = 0

            index_list = [(sn_name, log_time) for log_time in data["LogTime"]]
            data.index = pd.MultiIndex.from_tuples(index_list)
            return data
        return None

    def generate_and_save_data(self) -> NoReturn:
        """
        生成并保存负样本数据
        """
        data_all = self._get_data()
        feather.write_dataframe(
            data_all, os.path.join(self.train_data_path, "negative_train.csv")
        )


class TestDataGenerator(DataGenerator):
    @staticmethod
    def _split_dataframe(df: pd.DataFrame, chunk_size=2000000):
        """
        将 DataFrame 按照 chunk_size 进行切分
        :param df: 拆分前的 DataFrame
        :param chunk_size: chunk 大小
        :return: 切分后的 DataFrame, 每次返回一个 chunk
        """
        for start in range(0, len(df), chunk_size):
            yield df[start : start + chunk_size]

    def _process_file(self, sn_file: str) -> Union[pd.DataFrame, None]:
        """
        处理单个文件, 获取测试数据
        :param sn_file: 文件名
        :return: 处理后的 DataFrame
        """
        sn_name = os.path.splitext(sn_file)[0]
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
        生成并保存测试数据
        """
        data_all = self._get_data()
        for index, chunk in enumerate(self._split_dataframe(data_all)):
            feather.write_dataframe(
                chunk, os.path.join(self.test_data_path, f"res_{index}.csv")
            )


class Model(object):
    def __init__(self, config: Config):
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

    def train(self) -> NoReturn:
        """
        训练模型
        """
        train_pos = feather.read_dataframe(
            os.path.join(self.train_data_path, "positive_train.csv")
        )
        train_neg = feather.read_dataframe(
            os.path.join(self.train_data_path, "negative_train.csv")
        )
        train_all = pd.concat([train_pos, train_neg])
        train_all.drop("LogTime", axis=1, inplace=True)
        train_all = train_all.sort_index(axis=1)

        self.model.fit(train_all.drop(columns=["label"]), train_all["label"])

    def predict_proba(self) -> Dict[str, List]:
        """
        预测测试数据每个样本被预测为正类的概率, 并返回结果。这里返回的结果是一个 dict, key 为 sn_name, value 为预测结果
        :return: 每个样本被预测为正类的概率
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
                if p_s < 0.1:
                    continue
                sn = index_list[i][0]
                sn_t = datetime.fromtimestamp(index_list[i][1])
                result.setdefault(sn, [])
                result[sn].append((sn_t, p_s))
        return result

    def predict(self, threshold: int = 0.5) -> Dict[str, List]:
        """
        获得特定阈值下的预测结果,
        :param threshold: 阈值
        :return: 按照阈值筛选后的预测结果
        """
        result = self.predict_proba()

        # 将预测结果按照阈值进行筛选
        result = {
            sn: [int(sn_t.timestamp()) for sn_t, p_s in pred_list if p_s >= threshold]
            for sn, pred_list in result.items()
        }
        # 过滤空预测结果, 并将预测结果按照时间进行排序
        result = {
            sn: sorted(pred_list) for sn, pred_list in result.items() if pred_list
        }

        return result


if __name__ == "__main__":
    sn_type = "A"
    test_stage = 1

    if test_stage == 1:
        test_data_range: tuple = ("2024-06-01", "2024-08-01")
    else:
        test_data_range: tuple = ("2024-08-01", "2024-10-01")

    config = Config(
        data_path=os.path.join("D:/competition_data", f"type_{sn_type}"),
        feature_path=os.path.join(
            "D:/release_features/combined_sn_feature", f"type_{sn_type}"
        ),
        train_data_path=os.path.join(
            "D:/release_features/train_data", f"type_{sn_type}"
        ),
        test_data_path=os.path.join(
            "D:/release_features/test_data", f"type_{sn_type}_{test_stage}"
        ),
        test_data_range=test_data_range,
    )
    feature_factory = FeatureFactory(config)
    
    feature_factory.process_all_sn()

    positive_data_generator = PositiveDataGenerator(config)
    positive_data_generator.generate_and_save_data()

    negative_data_generator = NegativeDataGenerator(config)
    negative_data_generator.generate_and_save_data()

    test_data_generator = TestDataGenerator(config)
    test_data_generator.generate_and_save_data()

    model = Model(config)
    model.train()
    result = model.predict()

    submission = []
    for sn in result:
        for timestamp in result[sn]:
            submission.append([sn, timestamp, sn_type])
    submission = pd.DataFrame(submission, columns=['serial_number', 'prediction_timestamp', 'serial_number_type'])
    submission.to_csv('submission.csv', index=False, encoding='utf-8')

    print()
