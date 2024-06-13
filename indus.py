# Author: Mian Qin
# Date Created: 6/12/24
import numpy as np
from scipy.special import logsumexp

from utils import OPDataset
from wham import BinlessWHAM


def main():
    ...
    

if __name__ == "__main__":
    main()


class INDUS:
    def __init__(self, data: OPDataset):
        self.binlessWHAM = BinlessWHAM(data)
        self.data = data

    def __call__(self, column_names, target_column):
        self.binlessWHAM._1_preprocess_data(column_names)

    def integrate(self, column_names, target_columns):
        F_i = self.binlessWHAM.F_i
        Ui_Zj = self.binlessWHAM.Ui_Zj
        N_i = self.binlessWHAM.N_i
        coordinates = self.binlessWHAM.coordinates
        ln_C = -logsumexp(a=-logsumexp(a=F_i-self.data.beta*Ui_Zj, b=N_i, axis=0, keepdims=True), axis=1, keepdims=True)
        possible_N = list(range(0, 36))  # TODO: 所有出现过的N或用户输入
        ln_P = []
        for N in possible_N:
            delta = np.abs(coordinates[target_column] - N) < 0.01
            ln_PN = ln_C + logsumexp(a=-logsumexp(a=F_i-self.data.beta*Ui_Zj, b=N_i, axis=0, keepdims=True), b=delta, axis=1, keepdims=True)
            ln_P.append(ln_PN.item())
        ln_P = np.array(ln_P)
        return ln_P
