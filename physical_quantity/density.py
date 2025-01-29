# Author: Mian Qin
# Date Created: 2024/11/14
import numpy as np

from .prototype import TDependentQuantity


class IceDensity(TDependentQuantity):
    name = "rho_I"
    latex_name = r"\rho_I"
    unit = "mol/m^3"


def rho_I_real_water_func(T):
    """
    @Murray_2010_HomogeneousNucleationAmorphous
    """
    rho_I = -1.3103e-9 * T ** 3 + 3.8109e-7 * T ** 2 - 9.2592e-5 * T + 0.94040  # g/cm^3
    rho_I = rho_I / 18 * 1e6  # Convert into mol/m^3
    return rho_I


rho_I_real_water = IceDensity("RealWater", (200, 300), rho_I_real_water_func)  # TODO: valid range is incorrect


def rho_I_tip4p_ice_func(T):
    """
    @Thosar_2024_EngulfmentAntifreezeProteins
    """
    rho_I = 5e4
    return rho_I


rho_I_tip4p_ice = IceDensity("Tip4pIce", (200, 300), rho_I_tip4p_ice_func)  # TODO: valid range is incorrect


def main():
    rho_I_real_water.plot_T_dependent()
    rho_I_tip4p_ice.plot_T_dependent()
    IceDensity.plot_all()


if __name__ == "__main__":
    main()
