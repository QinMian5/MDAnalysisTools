# Author: Mian Qin
# Date Created: 2024/11/15
import numpy as np

from .prototype import PhysicalQuantity


class MeltingTemperature(PhysicalQuantity):
    name = "T_m"
    latex_name = "T_m"
    unit = "K"


T_m_real_water = MeltingTemperature("RealWater", 273.15)
T_m_tip4p_ice = MeltingTemperature("Tip4pIce", 272.2)


def main():
    MeltingTemperature.plot_all()


if __name__ == "__main__":
    main()
