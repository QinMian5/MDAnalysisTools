# Author: Mian Qin
# Date Created: 7/15/24
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants as c


class OPData:
    def __init__(self, data: pd.DataFrame, params: dict, relaxation_after_ramp=200):
        self._T = None
        self._beta = None
        self._df: pd.DataFrame = data
        t = self._df["t"].values
        time_step = np.mean(np.diff(t))
        self._time_step = time_step
        self._autocorr_time: float | None = None  # In ps
        self._independent_samples: int | None = None
        self._prd_start: float | None = None  # In ps
        self.params = params
        self.T = params["TEMPERATURE"]
        self._ramp_time = params["RAMP_TIME"]
        self._relaxation_time: float | None = None  # In ps
        self._prd_time = params["PRD_TIME"]
        # TODO: Determine t_drop

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value: int | float):
        self._T = value
        self._beta = 1 / (c.k * self._T)

    @property
    def beta(self):
        return self._beta

    def get_x_star_t(self, op):
        df = self.df_original
        t = df["t"].values
        x_star = self.params[op]["X_STAR"]
        x_star_init = self.params[op]["X_STAR_INIT"]
        ramp_time = self.params["RAMP_TIME"]
        k = (x_star - x_star_init) / ramp_time
        x_star_t = np.where(t < ramp_time, x_star_init + k * t, x_star)
        return x_star_t

    def df_in_range(self, t_start=None, t_end=None):
        if t_start is None:
            t_start = self._df["t"].min()
        if t_end is None:
            t_end = self._df["t"].max()
        index = (self._df["t"] >= t_start) & (self._df["t"] <= t_end)
        return self._df[index].copy()

    def df_before_t(self, t: float):
        return self.df_in_range(t_end=t)

    def df_after_t(self, t: float):
        return self.df_in_range(t_start=t)

    @property
    def df_after_ramp(self):
        return self.df_after_t(t=self._ramp_time)

    @property
    def df_prd(self):
        return self.df_after_t(t=self._prd_start)

    @property
    def df_bootstrap(self):
        df = self.df_prd
        resampled_df = df.sample(n=self.independent_samples, replace=True)
        return resampled_df

    @property
    def df_block_bootstrap(self):
        df = self.df_prd
        N_block = self.independent_samples
        block_size = int(len(df) / N_block)
        index_start = np.random.randint(0, len(df) - block_size + 1, size=(N_block,))
        df_list = [df.iloc[index:index + block_size] for index in index_start]
        df_bb = pd.concat(df_list, axis=0, ignore_index=True)
        return df_bb

    @property
    def df_original(self):
        return self._df.copy()

    @property
    def autocorr_time(self):
        return self._autocorr_time  # Unit: ps

    @autocorr_time.setter
    def autocorr_time(self, value: float):
        self._autocorr_time = value
        t = self.df_prd["t"].values
        t_tot = t[-1] - t[0]
        self._independent_samples = int(np.ceil(t_tot / self._autocorr_time))

    @property
    def relaxation_time(self):
        return self._relaxation_time

    @relaxation_time.setter
    def relaxation_time(self, value: float):
        self._relaxation_time = value
        self._prd_start = self._ramp_time + self._relaxation_time

    @property
    def prd_start(self):
        return self._prd_start

    @property
    def independent_samples(self):
        return self._independent_samples

    @property
    def time_step(self):
        return self._time_step

    def calculate_bias_potential(self, coordinates: dict[str, float | np.ndarray]) -> float | np.ndarray:
        """
            Calculate the bias potential (in kilojoules per mole) for a given position.
            Compatible with ufloat.

            :param coordinates: The coordinate (or numpy array of positions) at which to calculate the bias potential.
            :return: Bias potential (or numpy array of bias potentials).
        """
        sample_value = next(iter(coordinates.values()))
        total_bias_potential = np.zeros_like(sample_value)

        for variable, value in coordinates.items():
            if variable not in self.params:  # Unbiased
                continue
            kappa = self.params[variable].get("KAPPA", 0)
            x_star = self.params[variable].get("X_STAR", 0)
            phi = self.params[variable].get("PHI", 0)

            bias_potential = 0.5 * kappa * (value - x_star) ** 2 + phi * value  # kJ/mol
            total_bias_potential += bias_potential
            return total_bias_potential


class OPDataset(OrderedDict[str, OPData]):
    """
    A specialized ordered dictionary to store and manage data from umbrella sampling simulations.

    Each entry in this dictionary represents a single simulation condition. The key is the 'specifier'
    (normally the filename) associated with that simulation, and the value is a list consisting of
    a dict containing the bias parameters and a 'DataFrame' containing the simulation data.
    """

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dir = Path(data_dir)
        self.save_dir = self.data_dir.parent / "intermediate_result"
        self.save_dir.mkdir(exist_ok=True)

    @property
    def T(self):
        T = []
        for job_name, data in self.items():
            T.append(data.T)
        T = np.array(T).reshape(-1, 1)
        return T

    @property
    def beta(self):
        beta = []
        for job_name, data in self.items():
            beta.append(data.beta)
        beta = np.array(beta).reshape(-1, 1)
        return beta

    def calculate_bias_potential(self, coordinates: dict[str, float | np.ndarray], job_name: str) -> float | np.ndarray:
        """
            Calculate the bias potential for a given position.

            :param coordinates: The coordinate (or numpy array of positions) at which to calculate the bias potential.
            :param job_name: Job name.
            :return: The calculated bias potential (or numpy array of bias potentials).
        """
        bias_potential = self[job_name].calculate_bias_potential(coordinates)
        return bias_potential

    def update_autocorr_time(self, act_dict: dict[str, float]):
        for job_name, act in act_dict.items():
            self[job_name].autocorr_time = act

    def save_relaxation_time(self):
        for job_name, op_data in self.items():
            if op_data.relaxation_time is not None:
                save_path = self.save_dir / job_name / "relaxation_time.txt"
                save_path.parent.mkdir(exist_ok=True)
                with open(save_path, "w") as file:
                    file.write(f"{op_data.relaxation_time}\n")
                print(f"{job_name}: Saved relaxation time to {save_path}")

    def load_relaxation_time(self):
        for job_name, op_data in self.items():
            load_path = self.save_dir / job_name / "relaxation_time.txt"
            if load_path.exists():
                with open(load_path, "r") as file:
                    relaxation_time = float(file.read().strip())
                    op_data.relaxation_time = relaxation_time
                    # print(f"{job_name}: Loaded relaxation time from {load_path}")

    def save_act(self):
        for job_name, op_data in self.items():
            if op_data.autocorr_time is not None:
                save_path = self.save_dir / job_name / "act.txt"
                save_path.parent.mkdir(exist_ok=True)
                with open(save_path, "w") as file:
                    file.write(f"{op_data.autocorr_time}\n")
                print(f"{job_name}: Saved ACT to {save_path}")

    def load_act(self):
        for job_name, op_data in self.items():
            load_path = self.save_dir / job_name / "act.txt"
            if load_path.exists():
                with open(load_path, "r") as file:
                    act = float(file.read().strip())
                    op_data.autocorr_time = act
                    # print(f"{job_name}: Loaded ACT from {load_path}")


def load_dataset(data_dir: [str, Path], job_params: dict[str, dict], file_type: str,
                 column_names: list[str], column_types: dict[str, type]) -> OPDataset:
    """

    :param data_dir: Directory path containing data files.
    :param job_params: Job parameters.
    :param file_type: .csv or .out file.
    :param column_names: A list of strings specifying the column names of the data.
    :param column_types: A dictionary specifying the data types (int, float, etc.) of columns.
                         Only columns specified in column_types will be retained; all others will be disregarded.
    :return: An instance of UmbrellaSamplingDataset that contains the simulation data.
    """
    dataset = OPDataset(data_dir)
    data_dir = Path(data_dir)

    for job_name, params in job_params.items():
        if file_type == "csv":
            op_file_path = data_dir / job_name / "op_combined.csv"
            df = pd.read_csv(op_file_path, sep=',', header=0, names=column_names, comment='#')
        else:
            op_file_path = data_dir / job_name / "op.out"
            df = pd.read_csv(op_file_path, sep=r'\s+', header=None, names=column_names, comment='#')
        # Keep only columns present in column_types and drop others
        df = df.loc[:, df.columns.intersection(column_types.keys())]
        # Convert the types of the columns as specified in column_types
        for column_name, column_type in column_types.items():
            df[column_name] = df[column_name].astype(column_type)

        data = OPData(df, params)
        dataset[job_name] = data
    dataset.load_relaxation_time()
    dataset.load_act()
    return dataset


def main():
    ...


if __name__ == "__main__":
    main()

