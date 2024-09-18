# Author: Mian Qin
# Date Created: 6/4/24
import json

from pathlib import Path
import numpy as np
import scipy.constants as c
from scipy.integrate import cumulative_simpson
import pandas as pd
import matplotlib.pyplot as plt

from utils import convert_unit
from op_dataset import OPDataset


class SparseSampling:
    result_filename = "result_sparse_sampling.csv"

    def __init__(self, dataset: None | OPDataset, op: str):
        self.dataset = dataset
        self.op = op
        self.x: np.ndarray | None = None
        self.x_star: np.ndarray | None = None
        self.dF_lambda_dx_star: np.ndarray | None = None
        self.F_lambda: np.ndarray | None = None  # x_star
        self.F_nu_lambda: np.ndarray | None = None  # x
        self.U_lambda: np.ndarray | None = None  # x
        self.F_nu: np.ndarray | None = None  # x
        self._dF_nu_dx: np.ndarray | None = None
        self._sigma_dF_nu_dx: np.ndarray | None = None
        self.energy: np.ndarray | None = None  # x

    @property
    def dF_nu_dx(self):
        return self._dF_nu_dx, self._sigma_dF_nu_dx

    def _calculate_x(self) -> None:
        column_name = self.op
        x_list = []
        for job_name, data in self.dataset.items():
            op_values = data.df[column_name].values
            mean = op_values.mean()
            x_list.append(mean)
        self.x = np.array(x_list)

    def _calculate_dF_lambda_dx_star(self) -> None:
        column_name = self.op
        x_star_list = []
        df_lambda_dx_star_list = []
        for job_name, data in self.dataset.items():
            params = data.params[column_name]
            op_values = data.df[column_name].values
            assert params["TYPE"] in ["parabola"], f"Unknown bias potential type: {params['type']}"
            if params["TYPE"] == "parabola":
                op_center = params["CENTER"]
                kappa = params["KAPPA"]
                dF_lambda_dx_star = kappa * (op_center - op_values.mean())
                x_star_list.append(op_center)
                df_lambda_dx_star_list.append(dF_lambda_dx_star)
        self.x_star = np.array(x_star_list)
        self.dF_lambda_dx_star = np.array(df_lambda_dx_star_list)

    def _calculate_F_lambda(self) -> None:
        x_star = self.x_star
        dF_lambda_dx_star = self.dF_lambda_dx_star
        F_lambda = cumulative_simpson(dF_lambda_dx_star, x=x_star, initial=0)
        self.F_lambda = F_lambda

    def _calculate_F_nu_lambda(self) -> None:
        column_name = self.op
        F_nu_lambda_list = []
        for job_name, data in self.dataset.items():
            op_values = data.df[column_name].values
            std = op_values.std()

            # n_sigma = 0.1
            # mean = op_values.mean()
            # N_total = len(op_values)
            # N_center = len(op_values[(op_values <= mean + std * n_sigma) & (op_values >= mean - std * n_sigma)])
            # N_left = len(op_values[(op_values <= mean - std * n_sigma) & (op_values >= mean - 3 * std * n_sigma)])
            # N_right = len(op_values[(op_values <= mean + 3 * std * n_sigma) & (op_values >= mean + std * n_sigma)])
            # p_left = N_left / N_total
            # p_right = N_right / N_total

            F_nu_lambda = 0.5 * np.log(2 * np.pi * std ** 2) / data.beta
            F_nu_lambda = F_nu_lambda / 1000 * c.N_A  # To kJ/mol
            F_nu_lambda_list.append(F_nu_lambda)
        self.F_nu_lambda = np.array(F_nu_lambda_list)

    def _calculate_U_lambda(self) -> None:
        column_name = self.op
        x = self.x
        U_lambda_list = []
        for i, (job_name, data) in enumerate(self.dataset.items()):
            U_lambda = data.calculate_bias_potential({column_name: x[i]})
            U_lambda_list.append(U_lambda)
        self.U_lambda = np.array(U_lambda_list)

    def _calculate_F_nu(self) -> None:
        F_nu_lambda = self.F_nu_lambda
        U_lambda = self.U_lambda
        F_lambda = self.F_lambda

        F_nu = F_nu_lambda - U_lambda + F_lambda
        self.F_nu = F_nu

    def _calculate_dF_nu_dx(self) -> None:
        column_name = self.op
        dF_nu_lambda_dx_list = []
        dU_lambda_dx_list = []
        sigma_dU_lambda_dx_list = []
        for job_name, data in self.dataset.items():
            params = data.params[column_name]
            op_values = data.df[column_name].values
            assert params["TYPE"] in ["parabola"], f"Unknown bias potential type: {params['type']}"
            if params["TYPE"] == "parabola":
                op_center = params["CENTER"]
                kappa = params["KAPPA"]
                dF_nu_lambda_dx = 0  # Assume x_mean is the equilibrium position
                dF_nu_lambda_dx_list.append(dF_nu_lambda_dx)
                dU_lambda_dx = kappa * (op_values.mean() - op_center)
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

    def _calculate_energy(self) -> None:
        F_nu = self.F_nu
        energy = F_nu - F_nu.min()
        self.energy = energy

    def calculate(self):
        self._calculate_x()
        self._calculate_dF_lambda_dx_star()
        self._calculate_F_lambda()
        self._calculate_F_nu_lambda()
        self._calculate_U_lambda()
        self._calculate_F_nu()
        self._calculate_energy()
        # self._calculate_dF_nu_dx()

    def plot(self, save_fig: bool = True, save_dir=Path("./figure"), delta_mu=0.0):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        fig, ax = plt.subplots()
        x = self.x
        F_nu = self.F_nu
        line1 = ax.plot(x, convert_unit(F_nu) + delta_mu * x, "bo-")
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel(r"$\beta F + \Delta\mu N$")
        ax.tick_params(axis='y')

        # lines = [line1[0]]
        # labels = [line.get_label() for line in lines]
        # ax.legend(lines, labels)
        plt.title(rf"Sparse Sampling, $\Delta\mu = {delta_mu} k_BT$")
        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSampling_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def plot_different_DeltaT(self, save_fig: bool = True, save_dir=Path("./figure")):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        x = self.x
        F_nu = self.F_nu

        fig, ax = plt.subplots()
        ax.plot(x, convert_unit(F_nu), "o-", label=r"Raw data (at $300\ \mathrm{K}$)")
        for Delta_T in range(0, 105, 10):
            T_m = 271  # K
            Delta_H_m = 5.6  # kJ/mol
            Delta_mu = - Delta_H_m / T_m * (Delta_T + 300 - T_m)
            ax.plot(x, convert_unit(F_nu + Delta_mu * x, T=T_m-Delta_T), "o-", label=rf"${Delta_T}\ \mathrm{{K}}$")
        ax.set_xlabel(rf"$\lambda$")
        ax.set_ylabel(r"$G(\lambda;\Delta T)$")
        ax.tick_params(axis='y')

        # lines = [line1[0]]
        # labels = [line.get_label() for line in lines]
        ax.legend()
        plt.title(rf"Free Energy Profile at Different $\Delta T$")
        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSampling_{column_name}_DeltaT.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

        index = x <= 400
        x = x[index]
        F_nu = F_nu[index]

        fig, ax = plt.subplots()
        for Delta_T in range(0, 105, 5):
            T_m = 271  # K
            Delta_H_m = 5.6  # kJ/mol
            Delta_mu = - Delta_H_m / T_m * (Delta_T + 300 - T_m)
            ax.plot(x, convert_unit(F_nu + Delta_mu * x, T=T_m - Delta_T), "o-",
                            label=rf"${Delta_T}\ {{\rm K}}$")
        ax.set_xlabel(rf"$\lambda$")
        ax.set_ylabel(r"$G(\lambda;\Delta T)$")
        ax.tick_params(axis='y')

        # lines = [line1[0]]
        # labels = [line.get_label() for line in lines]
        ax.legend()
        plt.title(rf"Free Energy Profile at Different $\Delta T$ (Zoomed In)")
        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSampling_{column_name}_DeltaT_zoomed_in.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def plot_debug(self, save_fig: bool = True, save_dir=Path("./figure")):
        plt.style.use("presentation.mplstyle")
        column_name = self.op
        fig, ax1 = plt.subplots()
        ax1.set_title("Sparse Sampling (Debug)")
        ax1.set_xlabel(f"{column_name}")
        ax1.set_ylabel(r"$\beta F$", color="blue")
        ax1.tick_params(axis='y', colors="blue")
        x = self.x
        F_nu_lambda = self.F_nu_lambda
        line1 = ax1.plot(x, convert_unit(F_nu_lambda), "bo--", label=r"$\beta F_{\nu}^{\lambda}$")
        U_lambda = self.U_lambda
        line2 = ax1.plot(x, convert_unit(U_lambda), "bv--", label=r"$\beta U_{\lambda}$")

        ax2 = ax1.twinx()
        ax2.set_ylabel(r"$\beta dF_{\lambda} / dx$", color="red")
        ax2.tick_params(axis='y', colors="red")
        x_star = self.x_star
        dF_lambda_dx_star = self.dF_lambda_dx_star
        line4 = ax2.plot(x_star, convert_unit(dF_lambda_dx_star), "ro--", label=r"$\beta dF_{\lambda} / dx$")

        lines = [line1[0], line2[0], line4[0]]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels)

        if save_fig:
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"SparseSamplingDebug_{column_name}.png"
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
            print(f"Saved the figure to {save_path.resolve()}")
        else:
            plt.show()
        plt.close(fig)

    def save_result(self, save_dir=Path(".")):
        x = self.x
        x_star = self.x_star
        energy = self.energy
        dF_lambda_dx_star = self.dF_lambda_dx_star
        dF_nu_dx, sigma_dF_nu_dx = self.dF_nu_dx
        df = pd.DataFrame({"x": x, "x_star": x_star, "energy": energy, "dF_lambda_dx_star": dF_lambda_dx_star,
                           "dF_nu_dx": dF_nu_dx, "sigma_dF_nu_dx": sigma_dF_nu_dx})
        save_dir.mkdir(exist_ok=True)
        save_path = save_dir / self.result_filename
        df.to_csv(save_path)
        print(f"Saved the result to {save_path.resolve()}")

    def load_result(self, save_dir=Path(".")):
        save_path = save_dir / self.result_filename
        df = pd.read_csv(save_path)
        x = df["x"].values
        x_star = df["x_star"].values
        energy = df["energy"].values
        dF_lambda_dx_star = df["dF_lambda_dx_star"].values
        dF_nu_dx = df["dF_nu_dx"].values
        sigma_dF_nu_dx = df["sigma_dF_nu_dx"].values
        self.x = x
        self.x_star = x_star
        self.energy = energy
        self.dF_lambda_dx_star = dF_lambda_dx_star
        self._dF_nu_dx = dF_nu_dx
        self._sigma_dF_nu_dx = sigma_dF_nu_dx
        print(f"Loaded the result from {save_path.resolve()}")


def main():
    ...


if __name__ == "__main__":
    main()
