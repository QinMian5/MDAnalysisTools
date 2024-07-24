# Author: Mian Qin
# Date Created: 12/15/23

from pathlib import Path
import numpy as np
import pandas as pd
import scipy.constants as c
import autograd.numpy as agnp
import matplotlib.pyplot as plt

from utils import calculate_histogram_parameters, convert_unit
from op_dataset import OPDataset
from optimize import LBFGS, alogsumexp


class BinlessWHAM:
    result_filename = "result_wham.csv"

    def __init__(self, dataset: OPDataset, op: str):
        self.dataset = dataset
        self.op = op
        self.N_i: None | np.ndarray = None
        self.N_tot: None | int = None
        self.Ui_Zj: None | np.ndarray = None
        self.coordinates: None | dict[str, np.ndarray] = None
        self.F_i: None | np.ndarray = None
        self.energy: None | list[np.ndarray, np.ndarray] = None

    def _1_preprocess_data(self, column_names):
        N_i = []
        all_coord = []
        for _, data in self.dataset.items():
            df = data.df
            N_i.append(len(df))
            coord = df[column_names].values
            all_coord.append(coord)
        N_i = np.array(N_i).reshape(-1, 1)
        N_tot = N_i.sum()
        all_coord = np.concatenate(all_coord, axis=0)
        coordinates = {}
        for column_name, coord in zip(column_names, all_coord.T):
            coordinates[column_name] = coord
        Ui_Zj = []  # bias potential
        for job_name, data in self.dataset.items():
            bias = data.calculate_bias_potential(coordinates) * 1000 / c.N_A
            Ui_Zj.append(bias)
        Ui_Zj = np.stack(Ui_Zj, axis=0)
        self.N_i = N_i
        self.N_tot = N_tot
        self.Ui_Zj = Ui_Zj
        self.coordinates = coordinates

    def _2_maximum_likelihood_estimate(self):
        # LBFGS
        N_i = self.N_i
        N_tot = self.N_tot
        Ui_Zj = self.Ui_Zj
        F_i0 = np.zeros(N_i.shape[0])
        F_i = LBFGS(self.NLL, F_i0, args=(N_i, N_tot, Ui_Zj, self.dataset.beta), iprint=-1)
        F_i = F_i.reshape(-1, 1)
        self.F_i = F_i

    def _3_calculate_energy(self, column_name, num_bins=None, bin_width=None, bin_range=None):
        num_bins, bin_range = calculate_histogram_parameters(self.dataset, column_name, num_bins, bin_width, bin_range)
        N_i = self.N_i
        Ui_Zj = self.Ui_Zj
        F_i = self.F_i
        beta = self.dataset.beta
        Z_j = self.coordinates[column_name]
        W_j = 1 / (np.sum(N_i * np.exp(F_i - beta * Ui_Zj), axis=0))
        hist_wj, bin_edges = np.histogram(Z_j, bins=num_bins, range=bin_range, weights=W_j)
        p_wj = hist_wj / np.sum(hist_wj)
        bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
        energy = -1 / self.dataset.beta.mean() * np.log(p_wj) / 1000 * c.N_A  # To kJ/mol
        energy = energy - energy.min()
        self.energy = [bin_midpoints, energy]

    @staticmethod
    def NLL(F_i: np.ndarray, N_i: np.ndarray, N_tot: int, Ui_Zj: np.ndarray, beta):
        """
        Negative Log Likelihood of observing our dataset

        :param F_i: Free energy of the i-th simulation. Numpy array of shape (I, )
        :param N_i: Number of observations in the i-th simulation. Numpy array of shape (I, 1)
        :param N_tot: Total number of observations in all simulations. Int
        :param Ui_Zj: Bias potential all observations under the i-th simulation. Numpy array of shape (I, J)
        :param beta: 1 / (kB * T)
        :return:
        """
        F_i = F_i.reshape(-1, 1)
        F_i = F_i - F_i.mean()
        A = -agnp.sum(N_i / N_tot * F_i, axis=0, keepdims=True) + 1 / N_tot * agnp.sum(
            alogsumexp(a=F_i - beta * Ui_Zj, b=N_i / N_tot, axis=0, keepdims=True), axis=1, keepdims=True)
        return A

    def calculate(self, column_names):
        column_name = self.op
        self._1_preprocess_data(column_names)
        self._2_maximum_likelihood_estimate()
        self._3_calculate_energy(column_name)

    def plot(self, save_fig=True, save_dir=Path("./figure")):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        x, energy = self.energy
        fig, ax = plt.subplots()
        ax.plot(x, convert_unit(energy), label="BinlessWHAM")
        ax.set_title(f"Binless WHAM")
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel(r"$\beta F$")
        ax.legend()
        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"WHAM_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def save_result(self, save_dir=Path(".")):
        x, energy = self.energy
        df = pd.DataFrame({"x": x, "energy": energy})
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / self.result_filename
        df.to_csv(save_path)
        print(f"Saved the result to {save_path.resolve()}")

    def load_result(self, save_dir=Path(".")):
        save_path = save_dir / self.result_filename
        df = pd.read_csv(save_path)
        x = df["x"].values
        energy = df["energy"].values
        self.energy = [x, energy]
        print(f"Loaded the result from {save_path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
