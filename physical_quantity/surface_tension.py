# Author: Mian Qin
# Date Created: 2024/11/14
import numpy as np

if __name__ == "__main__":
    from prototype import TDependentQuantity
    from physical_quantity import T_m_real_water, T_m_tip4p_ice
else:
    from .prototype import TDependentQuantity
    from .melting_temperature import T_m_real_water, T_m_tip4p_ice


class SurfaceTension(TDependentQuantity):
    name = "gamma_IL"
    latex_name = r"\gamma_{IL}"
    unit = "J/m^2"


def gamma_IL_real_water_func(T):
    """
    @Koop_2016_PhysicallyConstrainedClassical
    """
    T_m = T_m_real_water.value  # K
    Delta_H_m_h = (6.008 + 0.03616 * (T - T_m) - 3.9479e-4 * (T - T_m) ** 2 - 1.6248e-5 * (T - T_m) ** 3 - 3.2563e-7 * (
                T - T_m) ** 4) * 1e3  # J/mol, Enthalpy of melting for hexagonal ice
    Delta_H_sd_h = 0.155e3  # J/mol, Enthalpy difference between stacking disordered and hexagonal ice
    Delta_H_m_sd = Delta_H_m_h - Delta_H_sd_h
    Delta_H_m_sd_Tr = 4.1776e3  # J/mol, Delta_H_m_sd at Tr
    sigma_sd_l_Tr = 18.505e-3  # J/m^2
    sigma_sd_l = Delta_H_m_sd / Delta_H_m_sd_Tr * sigma_sd_l_Tr
    return sigma_sd_l


gamma_IL_real_water = SurfaceTension("RealWater", (225, 273.15), gamma_IL_real_water_func)


def gamma_IL_tip4p_ice_func(T):
    """
    @Espinosa_2014_HomogeneousIceNucleation
    """
    T_m = T_m_tip4p_ice.value
    gamma = (30.8 - 0.25 * (T_m - T)) * 1e-3  # J/m^2
    return gamma


gamma_IL_tip4p_ice = SurfaceTension("Tip4pIce", (230, 300), gamma_IL_tip4p_ice_func)  # TODO: valid range is incorrect


def main():
    gamma_IL_real_water.plot_T_dependent()
    gamma_IL_tip4p_ice.plot_T_dependent()
    SurfaceTension.plot_all()


if __name__ == "__main__":
    main()
