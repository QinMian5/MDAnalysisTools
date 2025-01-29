# Author: Mian Qin
# Date Created: 6/4/24
import json
from pathlib import Path

import numpy as np
import scipy.constants as c
from scipy.integrate import cumulative_simpson
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties import ufloat
import uncertainties.unumpy as unp

from utils import convert_unit
from utils_plot import create_fig_ax, save_figure, plot_with_error_bar
from op_dataset import OPDataset


class SparseSampling:
    data_to_save = ["x_u", "F_nu_u"]
    def __init__(self, dataset: None | OPDataset, op: str):
        self.dataset = dataset
        self.op = op
        self.x_u: np.ndarray | None = None  # average x
        self.x_star: np.ndarray | None = None
        self.dF_lambda_dx_star_u: np.ndarray | None = None
        self.F_lambda_u: np.ndarray | None = None  # x_star
        self.F_nu_lambda_u: np.ndarray | None = None  # x
        self.U_lambda_u: np.ndarray | None = None  # x
        self.F_nu_u: np.ndarray | None = None  # x, free energy
        self._dF_nu_dx: np.ndarray | None = None
        self._sigma_dF_nu_dx: np.ndarray | None = None
        self.save_dir = self.dataset.save_dir / "sparse_sampling"

    @property
    def dF_nu_dx(self):
        return self._dF_nu_dx, self._sigma_dF_nu_dx

    def calculate(self):
        self.__calculate_dF_lambda_dx_star()
        self.__calculate_F_lambda()
        self.__calculate_F_nu_lambda()
        self.__calculate_U_lambda()
        self.__calculate_F_nu()
        # self._calculate_dF_nu_dx()

    def __calculate_dF_lambda_dx_star(self) -> None:
        op = self.op
        x_list = []
        x_star_list = []
        df_lambda_dx_star_list = []
        for job_name, op_data in self.dataset.items():
            params = op_data.params[op]
            N_ind = op_data.independent_samples
            op_values = op_data.df_prd[op].values
            x = op_values.mean()
            s_x = np.std(op_values, ddof=1) / np.sqrt(N_ind)
            x_u = ufloat(x, s_x)
            x_star = params.get("X_STAR", params.get("STAR"))
            kappa = params["KAPPA"]
            dF_lambda_dx_star_u = kappa * (x_star - x_u)
            x_list.append(x_u)
            x_star_list.append(x_star)
            df_lambda_dx_star_list.append(dF_lambda_dx_star_u)
        self.x_u = np.array(x_list)
        self.x_star = np.array(x_star_list)
        self.dF_lambda_dx_star_u = np.array(df_lambda_dx_star_list)

    def __calculate_F_lambda(self, mc_samples=100000) -> None:
        x_star = self.x_star
        dF_lambda_dx_star_u = self.dF_lambda_dx_star_u
        dF_lambda_dx_star = unp.nominal_values(dF_lambda_dx_star_u)
        s_dF_lambda_dx_star = unp.std_devs(dF_lambda_dx_star_u)
        # Error Analysis by Monte-Carlo Simulation
        sample = np.random.normal(dF_lambda_dx_star, s_dF_lambda_dx_star, (mc_samples, dF_lambda_dx_star_u.shape[0]))
        res = cumulative_simpson(sample, x=x_star, initial=0)
        F_lambda = np.mean(res, axis=0)
        s_F_lambda = np.std(res, axis=0)
        F_lambda_u = unp.uarray(F_lambda, s_F_lambda)
        self.F_lambda_u = F_lambda_u

    def __calculate_F_nu_lambda(self) -> None:
        op = self.op
        F_nu_lambda_list = []
        for job_name, op_data in self.dataset.items():
            op_values = op_data.df_prd[op].values
            std = op_values.std()

            F_nu_lambda = 0.5 * np.log(2 * np.pi * std ** 2) / op_data.beta
            F_nu_lambda = F_nu_lambda / 1000 * c.N_A  # To kJ/mol
            F_nu_lambda_u = ufloat(F_nu_lambda, 0)
            F_nu_lambda_list.append(F_nu_lambda_u)
        self.F_nu_lambda_u = np.array(F_nu_lambda_list)

    def __calculate_U_lambda(self) -> None:
        column_name = self.op
        x_u = self.x_u
        U_lambda_list = []
        for i, (job_name, data) in enumerate(self.dataset.items()):
            U_lambda_u = data.calculate_bias_potential({column_name: x_u[i]})
            U_lambda_list.append(U_lambda_u)
        self.U_lambda_u = np.array(U_lambda_list)

    def __calculate_F_nu(self) -> None:
        F_nu_lambda_u = self.F_nu_lambda_u
        U_lambda_u = self.U_lambda_u
        F_lambda_u = self.F_lambda_u

        F_nu_u = F_nu_lambda_u - U_lambda_u + F_lambda_u
        F_nu_u -= F_nu_u[0]
        self.F_nu_u = F_nu_u

    def __calculate_dF_nu_dx(self) -> None:
        column_name = self.op
        dF_nu_lambda_dx_list = []
        dU_lambda_dx_list = []
        sigma_dU_lambda_dx_list = []
        for job_name, data in self.dataset.items():
            params = data.params[column_name]
            op_values = data.df_prd[column_name].values
            assert params["TYPE"] in ["parabola"], f"Unknown bias potential type: {params['type']}"
            if params["TYPE"] == "parabola":
                op_star = params["STAR"]
                kappa = params["KAPPA"]
                dF_nu_lambda_dx = 0  # Assume x_mean is the equilibrium position
                dF_nu_lambda_dx_list.append(dF_nu_lambda_dx)
                dU_lambda_dx = kappa * (op_values.mean() - op_star)
                dU_lambda_dx_list.append(dU_lambda_dx)

                N_independent_samples = data.independent_samples
                sigma_dU_lambda_dx = kappa * op_values.std() / np.sqrt(N_independent_samples)
                sigma_dU_lambda_dx_list.append(sigma_dU_lambda_dx)
        dF_nu_lambda_dx = np.array(dF_nu_lambda_dx_list)
        dU_lambda_dx = np.array(dU_lambda_dx_list)
        dF_nu_dx = dF_nu_lambda_dx - dU_lambda_dx
        sigma_dU_lambda_dx = np.array(sigma_dU_lambda_dx_list)
        sigma_dF_nu_dx = sigma_dU_lambda_dx
        self._dF_nu_dx = dF_nu_dx
        self._sigma_dF_nu_dx = sigma_dF_nu_dx

    def plot_free_energy_plot_line(self, ax, delta_mu=None, T=None, label=None, x_range=None):
        if T is None:
            T = self.dataset.T.mean()
        if delta_mu is None:
            delta_mu = 0
        x_u = self.x_u
        x = unp.nominal_values(x_u)
        F_nu_u = self.F_nu_u
        if x_range is not None:
            index = (x_range[0] <= x) & (x <= x_range[1])
            x = x[index]
            x_u = x_u[index]
            F_nu_u = F_nu_u[index]

        line = plot_with_error_bar(ax, x, convert_unit(F_nu_u + delta_mu * x_u, T=T), "o-", label=label)
        return line

    def plot_free_energy(self, save_dir=Path("./figure")):
        op = self.op
        title = "Free Energy"
        x_label = f"{op}"
        y_label = fr"$\beta F$"
        fig, ax = create_fig_ax(title, x_label, y_label)
        # ax.tick_params(axis='y')
        self.plot_free_energy_plot_line(ax)

        save_path = save_dir / f"sparse_sampling_free_energy.png"
        save_figure(fig, save_path)
        plt.close(fig)

    def plot_different_DeltaT(self, save_dir=Path("./figure")):
        T_m = 271  # K
        Delta_H_m = 5.6  # kJ/mol
        T_sim = np.mean(self.dataset.T)

        title = fr"Free Energy Profile at Different $\Delta T$"
        x_label = fr"$\lambda$"
        y_label = fr"$G(\lambda;\Delta T) (kT)$"
        fig, ax = create_fig_ax(title, x_label, y_label)
        # ax.tick_params(axis='y')
        self.plot_free_energy_plot_line(ax, label=r"Raw data (at $300\ \mathrm{K}$)")

        for Delta_T in range(0, 105, 10):
            T = T_m - Delta_T
            Delta_mu = - Delta_H_m / T_m * (Delta_T + T_sim - T_m)
            label = fr"${Delta_T}\ \mathrm{{K}}$"
            self.plot_free_energy_plot_line(ax, delta_mu=Delta_mu, T=T, label=label)

        ax.legend()
        save_path = save_dir / f"sparse_sampling_DeltaT.png"
        save_figure(fig, save_path)
        plt.close(fig)

        title = fr"Free Energy Profile at Different $\Delta T$ (Zoomed In)"
        x_label = fr"$\lambda$"
        y_label = fr"$G(\lambda;\Delta T)$ (kT)"
        fig, ax = create_fig_ax(title, x_label, y_label)
        # ax.tick_params(axis='y')
        for Delta_T in range(0, 105, 5):
            T = T_m - Delta_T
            Delta_mu = - Delta_H_m / T_m * (Delta_T + T_sim - T_m)
            label = fr"${Delta_T}\ {{\rm K}}$"
            x_range = [0, 400]
            self.plot_free_energy_plot_line(ax, delta_mu=Delta_mu, T=T, label=label, x_range=x_range)

        ax.legend()
        save_path = save_dir / f"sparse_sampling_DeltaT_zoomed_in.png"
        save_figure(fig, save_path)
        plt.close(fig)

    def plot_dF_lambda_dx_star_plot_line(self, ax, label=None, **kwargs):
        x_star = self.x_star
        dF_lambda_dx_star_u = self.dF_lambda_dx_star_u
        line = plot_with_error_bar(ax, x_star, convert_unit(dF_lambda_dx_star_u), "o--", label=label, **kwargs)
        return line

    def plot_detail(self, save_dir=Path("./figure")):
        op = self.op
        title = "Sparse Sampling (Detail)"
        x_label = f"{op}"
        y_label = r"$\beta U_\lambda$"
        fig, ax1 = create_fig_ax(title, x_label, y_label)
        ax1.tick_params(axis='y', colors="blue")
        x_u = self.x_u
        x = unp.nominal_values(x_u)
        U_lambda_u = self.U_lambda_u
        line1 = plot_with_error_bar(ax1, x, convert_unit(U_lambda_u), "bo--", label=r"$\beta U_{\lambda}$")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\beta dF_{\lambda} / dx$")
        ax2.tick_params(axis='y', colors="red")
        line2 = self.plot_dF_lambda_dx_star_plot_line(ax2, label=r"$\beta dF_{\lambda} / dx$", color="red")

        lines = [line1, line2]
        labels = [str(line.get_label()) for line in lines]
        ax1.legend(lines, labels)

        save_path = save_dir / f"sparse_sampling_detail.png"
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
