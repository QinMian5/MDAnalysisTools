# Author: Mian Qin
# Date Created: 12/15/23
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.constants as c
import autograd.numpy as agnp
import matplotlib.pyplot as plt
from statsmodels.sandbox.distributions.gof_new import bootstrap
from uncertainties import ufloat
import uncertainties.unumpy as unp

from utils import calculate_histogram_parameters, convert_unit
from op_dataset import OPDataset
from optimize import LBFGSB, newton_raphson, alogsumexp
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar


class BinlessWHAM:
    data_to_save = ["bin_midpoint", "energy"]
    def __init__(self, dataset: OPDataset, op_in: list[str], op_out: str, num_bins=None, bin_width=None, bin_range=None):
        self.dataset = dataset
        self.op_in = op_in
        self.op_out = op_out
        self.num_bins = num_bins
        self.bin_width = bin_width
        self.bin_range = bin_range
        self.save_dir = self.dataset.save_dir / "wham"

        self.N_i: None | np.ndarray = None
        self.N_tot: None | int = None
        self.Ui_Zj: None | np.ndarray = None
        self.coordinates: None | dict[str, np.ndarray] = None
        self.F_i: None | np.ndarray = None
        self.energy: None | np.ndarray = None

        num_bins, bin_range = calculate_histogram_parameters(self.dataset, self.op_out, self.num_bins, self.bin_width, self.bin_range)
        _, bin_edges = np.histogram([], bins=num_bins, range=bin_range)
        self.bin_midpoint = (bin_edges[1:] + bin_edges[:-1]) / 2

    def calculate(self, with_uncertainties=False, n_iter=1000):
        energy = self.wham(bootstrap=False)
        if with_uncertainties:
            energy_list = []
            for i in tqdm(range(n_iter)):
                energy_bootstrap = self.wham(bootstrap=True)
                shift = np.mean(energy - energy_bootstrap)
                energy_bootstrap_shifted = energy_bootstrap + shift
                energy_list.append(energy_bootstrap_shifted)
            energy_array = np.stack(energy_list, axis=0)
            energy_mean = np.mean(energy_array, axis=0)
            energy_std = np.std(energy_array, axis=0, ddof=1)
            energy = unp.uarray(energy_mean, energy_std)
        self.energy = energy

    def wham(self, bootstrap=False):
        op_in = self.op_in
        N_i = []
        all_coord = []
        for _, op_data in self.dataset.items():
            df = op_data.df_block_bootstrap if bootstrap else op_data.df_prd
            N_i.append(len(df))
            coord = df[op_in].values
            all_coord.append(coord)
        N_i = np.array(N_i).reshape(-1, 1)
        N_tot = N_i.sum()
        all_coord = np.concatenate(all_coord, axis=0)
        coordinates = {}
        for column_name, coord in zip(op_in, all_coord.T):
            coordinates[column_name] = coord
        Ui_Zj = []  # bias potential
        for job_name, op_data in self.dataset.items():
            bias = op_data.calculate_bias_potential(coordinates) * 1000 / c.N_A  # to J
            Ui_Zj.append(bias)
        Ui_Zj = np.stack(Ui_Zj, axis=0)
        # LBFGS
        if self.F_i is None:
            F_i0 = np.zeros(N_i.shape[0])
        else:
            F_i0 = self.F_i.flatten()
        F_i_temp = LBFGSB(self.NLL, F_i0, args=(N_i, N_tot, Ui_Zj, self.dataset.beta), iprint=-1)
        F_i = F_i_temp
        # F_i = newton_raphson(self.NLL, F_i_temp, args=(N_i, N_tot, Ui_Zj, self.dataset.beta),)
        F_i = F_i.reshape(-1, 1)
        F_i = F_i - F_i.mean()
        self.F_i = F_i

        num_bins, bin_range = calculate_histogram_parameters(self.dataset, self.op_out, self.num_bins, self.bin_width, self.bin_range)
        beta = self.dataset.beta
        Z_j = coordinates[self.op_out]
        W_j = 1 / (np.sum(N_i * np.exp(F_i - beta * Ui_Zj), axis=0))
        hist_wj, bin_edges = np.histogram(Z_j, bins=num_bins, range=bin_range, weights=W_j)
        p_wj = hist_wj / np.sum(hist_wj)
        energy = -1 / self.dataset.beta.mean() * np.log(p_wj) / 1000 * c.N_A  # To kJ/mol
        energy = energy - energy.min()
        return energy

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

    def plot_free_energy_plot_line(self, ax, delta_mu=None, T=None, label=None, x_range=None):
        if T is None:
            T = self.dataset.T.mean()
        if delta_mu is None:
            delta_mu = 0
        x = self.bin_midpoint
        energy = self.energy
        if x_range is not None:
            index = (x_range[0] <= x) & (x <= x_range[1])
            x = x[index]
            energy = energy[index]

        line = plot_with_error_bar(ax, x, convert_unit(energy + delta_mu * x, T=T), "o-", label=label)
        return line

    def plot_free_energy(self, save_dir=Path("./figure")):
        op = self.op_out
        title = "Free Energy"
        x_label = f"{op}"
        y_label = fr"$\beta F$"
        fig, ax = create_fig_ax(title, x_label, y_label)
        self.plot_free_energy_plot_line(ax)

        save_path = save_dir / f"wham_free_energy_{op}.png"
        save_figure(fig, save_path)
        plt.close(fig)

    def save_result(self):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name in self.data_to_save:
            array = getattr(self, name)
            save_path = self.save_dir / f"{name}.npy"
            np.save(save_path, array)
            print(f"Saved self.{name} to {save_path}.")

    def load_result(self):
        for name in self.data_to_save:
            save_path = self.save_dir / f"{name}.npy"
            array = np.load(save_path, allow_pickle=True)
            setattr(self, name, array)
            print(f"Loaded self.{name} from {save_path}.")


def main():
    ...


if __name__ == "__main__":
    main()
