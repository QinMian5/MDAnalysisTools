# Author: Mian Qin
# Date Created: 2025/1/15
import numpy as np
import scipy.constants as c

if __name__ == "__main__":
    from prototype import TDependentQuantity
else:
    from .prototype import TDependentQuantity


class DiffusionCoefficient(TDependentQuantity):
    name = "D"
    latex_name = r"D"
    unit = "m^2/s"


def D_real_water_PL_func(T):
    """
    @Koop_2016_PhysicallyConstrainedClassical Table 5
    """
    D_star = 8.3175e-10  # m^2 s^-1 K^-0.5
    T_s = 215.45  # K
    gamma = 1.9188
    D = D_star * np.sqrt(T) * (T / T_s - 1) ** gamma
    return D


D_real_water_PL = DiffusionCoefficient("RealWaterPL", (237, 317), D_real_water_PL_func)


def D_real_water_VFT_func(T):
    """
    @Koop_2016_PhysicallyConstrainedClassical Table 6
    """
    D0 = 9.6307e-8  # m^2 s^-1
    T0 = 148.0  # K
    B = 560.96  # K
    D = D0 * np.exp(-B / (T - T0))
    return D


D_real_water_VFT = DiffusionCoefficient("RealWaterVFT", (237, 317), D_real_water_VFT_func)


def main():
    D_real_water_PL.plot_T_dependent()
    D_real_water_VFT.plot_T_dependent()
    DiffusionCoefficient.plot_all()


if __name__ == "__main__":
    main()
