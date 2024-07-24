# Author: Mian Qin
# Date Created: 7/15/24
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import constants as c


class OPData:
    def __init__(self, data: pd.DataFrame, params: dict):
        self._T = None
        self._beta = None
        self._df: pd.DataFrame = data
        t = self._df["t"].values
        time_step = np.mean(t[1:] - t[:-1])
        self._time_step = time_step
        self._autocorr_time: float | None = None  # In units of index
        self._independent_samples: int | None = None
        self.df = self._df.copy()
        self.params = params
        self.T = params["TEMPERATURE"]

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
    def autocorr_time(self):
        return self._autocorr_time

    @autocorr_time.setter
    def autocorr_time(self, value: float):
        self._autocorr_time = value
        N = len(self._df)
        self._independent_samples = int(np.ceil(N / self._autocorr_time))

    @property
    def independent_samples(self):
        return self._independent_samples

    @property
    def time_step(self):
        return self._time_step

    def drop_before(self, t: int | float):
        new_df = self._df[self._df["t"] >= t].copy()
        # new_df["t"] = new_df["t"] - t
        self._df = new_df
        self.df = self
        self.df = self._df

    # def resample(self, n=None, replace=True):
    #     if n is None:
    #         if self._independent_samples is None:
    #             raise ValueError("Auto-correlation time must be set before sampling.")
    #         n = self._independent_samples
    #     self.df = self._df.sample(n, replace=replace, axis=0)
    #
    # def restore(self):
    #     self.df = self._df.copy()

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
                center = self.params[variable]["CENTER"]
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def update_autocorr_time(self, tau_dict: dict[str, float]):
        for job_name, tau in tau_dict.items():
            self[job_name].autocorr_time = tau


def main():
    ...


if __name__ == "__main__":
    main()
