# Author: Mian Qin
# Date Created: 7/15/24
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants as c


class OPData:
    def __init__(self, data: pd.DataFrame, params: dict):
        self._T = None
        self._beta = None
        self._drop_before = 0
        self._df: pd.DataFrame = data
        t = self._df["t"].values
        time_step = np.mean(t[1:] - t[:-1])
        self._time_step = time_step
        self._autocorr_time: float | None = None  # In units of index
        self._independent_samples: int | None = None
        self.params = params
        self.T = params["TEMPERATURE"]
        self.ramp_time = params["RAMP_TIME"]
        self.prd_time = params["PRD_TIME"]
        self.drop_before(self.ramp_time + 200)

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

    @property
    def df(self):
        return self._df[self._df["t"] >= self._drop_before].copy()

    @property
    def initial_df(self):
        return self._df.copy()

    @property
    def autocorr_time(self):
        return self._autocorr_time  # Unit: ps

    @autocorr_time.setter
    def autocorr_time(self, value: float):
        self._autocorr_time = value
        t = self.df["t"].values
        t_tot = t[-1] - t[0]
        self._independent_samples = int(np.ceil(t_tot / self._autocorr_time))

    @property
    def independent_samples(self):
        return self._independent_samples

    @property
    def time_step(self):
        return self._time_step

    def drop_before(self, t: int | float):
        self._drop_before = t

    def calculate_bias_potential(self, coordinates: dict[str, float | np.ndarray]) -> float | np.ndarray:
        """
            Calculate the bias potential (in kilojoules per mole) for a given position.

            :param coordinates: The coordinate (or numpy array of positions) at which to calculate the bias potential.
            :return: Bias potential (or numpy array of bias potentials).
        """
        sample_value = next(iter(coordinates.values()))
        if isinstance(sample_value, float) or isinstance(sample_value, int):
            total_bias_potential = 0.0
        elif isinstance(sample_value, np.ndarray):
            total_bias_potential = np.zeros_like(sample_value)
        else:
            raise ValueError(f"Coordinates must either be float(int) or np.ndarray, got {type(sample_value)}")

        for variable, value in coordinates.items():
            if variable not in self.params:  # Unbiased
                continue
            bias_type = self.params[variable]["TYPE"]
            if bias_type == "parabola":
                kappa = self.params[variable]["KAPPA"]
                center = self.params[variable]["STAR"]
                # bias_potential = 0.5 * kappa * (value - center) ** 2 * 1000 / c.N_A
                bias_potential = 0.5 * kappa * (value - center) ** 2
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

    def drop_before(self, t):
        for name, data in self.items():
            data.drop_before(t)

    def update_autocorr_time(self, act_dict: dict[str, float]):
        for job_name, act in act_dict.items():
            self[job_name].autocorr_time = act

    def save_act(self):
        for job_name, op_data in self.items():
            if op_data.autocorr_time is not None:
                save_path = self.data_dir / job_name / "act.txt"
                with open(save_path, "w") as file:
                    file.write(f"{op_data.autocorr_time}\n")
                print(f"{job_name}: Saved ACT to {save_path}")

    def load_act(self):
        for job_name, op_data in self.items():
            load_path = self.data_dir / job_name / "act.txt"
            if load_path.exists():
                with open(load_path, "r") as file:
                    act = float(file.read().strip())
                    op_data.autocorr_time = act
                    print(f"{job_name}: Loaded ACT from {load_path}")


def load_dataset(data_dir: [str, Path], job_params: dict[str, dict],
                 column_names: list[str], column_types: dict[str, type]) -> OPDataset:
    """

    :param data_dir: Directory path containing data files.
    :param job_params: Job parameters.
    :param column_names: A list of strings specifying the column names of the data.
    :param column_types: A dictionary specifying the data types (int, float, etc.) of columns.
                         Only columns specified in column_types will be retained; all others will be disregarded.
    :return: An instance of UmbrellaSamplingDataset that contains the simulation data.
    """
    dataset = OPDataset(data_dir)
    data_dir = Path(data_dir)

    for job_name, params in job_params.items():
        # Read the file into a DataFrame, using whitespace as delimiter
        df = pd.read_csv(data_dir / job_name / "op.out", sep=r'\s+', header=None, names=column_names, comment='#')
        # Keep only columns present in column_types and drop others
        df = df.loc[:, df.columns.intersection(column_types.keys())]
        # Convert the types of the columns as specified in column_types
        for column_name, column_type in column_types.items():
            df[column_name] = df[column_name].astype(column_type)

        data = OPData(df, params)
        dataset[job_name] = data
    dataset.load_act()
    return dataset


def main():
    ...


if __name__ == "__main__":
    main()

