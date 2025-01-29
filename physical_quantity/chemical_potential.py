# Author: Mian Qin
# Date Created: 2024/11/14
import numpy as np

from .prototype import TDependentQuantity
from .melting_temperature import T_m_real_water, T_m_tip4p_ice


class ChemicalPotential(TDependentQuantity):
    name = "DeltaMu"
    latex_name = r"\Delta\mu"
    unit = "J/mol"


def delta_mu_real_water_func(T):
    """
    @Koop_2000_WaterActivityDeterminant
    """
    return 210368 + 131.438 * T - 3.32373e6 / T - 41729.1 * np.log(T)


delta_mu_real_water = ChemicalPotential("RealWater", (150, 273), delta_mu_real_water_func)


def delta_mu_tip4p_ice_func(T):
    """
    @Thosar_2024_EngulfmentAntifreezeProteins
    """
    Delta_h = 5.6e3  # J/mol
    T_m = T_m_tip4p_ice.value  # K
    Delta_mu = Delta_h / T_m * (T - T_m)
    return Delta_mu


delta_mu_tip4p_ice = ChemicalPotential("Tip4pIce", (100, 300), delta_mu_tip4p_ice_func)  # TODO: valid range is incorrect


def main():
    delta_mu_real_water.plot_T_dependent()
    delta_mu_tip4p_ice.plot_T_dependent()
    ChemicalPotential.plot_all()


if __name__ == "__main__":
    main()
