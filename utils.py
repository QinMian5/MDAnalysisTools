# Author: Mian Qin
# Date Created: 2/4/24
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import scipy.constants as c


class OPData:
    def __init__(self, data: pd.DataFrame, params: dict):
        self._T = None
        self._beta = None
        self.df: pd.DataFrame = data
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

    def drop_before(self, t: int | float):
        new_df = self.df[self.df["t"] >= t]
        self.df = new_df

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
            # TODO: Other bias_type
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
    dataset = OPDataset()
    data_dir = Path(data_dir)

    for job_name, params in job_params.items():
        # Read the file into a DataFrame, using whitespace as delimiter
        df = pd.read_csv(data_dir / job_name / "op.out", sep=r'\s+', header=None, names=column_names, comment='#')
        # Keep only columns present in column_types and drop others
        # df = df.loc[:, df.columns.intersection(column_types.keys())]
        # # Convert the types of the columns as specified in column_types
        # for column_name, column_type in column_types.items():
        #     df[column_name] = df[column_name].astype(column_type)

        data = OPData(df, params)
        dataset[job_name] = data
    return dataset


def calculate_histogram_parameters(dataset: OPDataset, column_name, num_bins: int = None,
                                   bin_width: float = None, bin_range: tuple[float, float] = None):
    """
        Calculate parameters for histogram plotting: num_bins and bin_range.

        :param dataset: An instance of the UmbrellaSamplingDataset that stores the simulation data.
        :param column_name: Name of the column to compute the histogram for.
        :param num_bins: Number of bins. If not specified, calculates based on bin_width or default strategy.
        :param bin_width: The bin width for the histogram. Used if num_bins is not specified.
        :param bin_range: A tuple specifying the minimum and maximum range of the bins.
                          If not specified, uses the minimum and maximum range in the data.
        :return: A tuple containing calculated num_bins and bin_range.
    """
    if bin_range is None:
        column_min = min(data.df[column_name].min() for data in dataset.values())
        column_max = max(data.df[column_name].max() for data in dataset.values())
        bin_range = (column_min, column_max)

    if num_bins is None and bin_width is None:
        total_points = sum(len(data.df[column_name]) for data in dataset.values())
        # The default strategy for choosing num_bins ensures an average of 1000 points per bin.
        num_bins = max(total_points // 1000, 1)
    elif bin_width is not None:
        # Calculate num_bins using bin_width if num_bins is not explicitly provided
        num_bins = int((bin_range[1] - bin_range[0]) / bin_width)

    return num_bins, bin_range


def main():
    def filename2bias_params(filename):
        param = filename.split('_')[1][:-4]
        return {"x": param}

    def filename2order(filename):
        order = float(filename.split('_')[1][:-4])
        return order

    data = load_dataset(
        data_dir=DATA_DIR,
        column_names=["t", "x", "y", "z"],
        column_types={"t": float, "x": float},
    )
    print(data)


if __name__ == "__main__":
    DATA_DIR = Path("/Users/qinmian/Data_unsync/testdata/ModelPotential1d/")
    main()
