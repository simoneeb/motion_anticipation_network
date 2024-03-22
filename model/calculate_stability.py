import numpy as np
import pickle
from connectivity import connectivity
import matplotlib.pyplot as plt



net_name = 'Reciporcal/Reciporcal_fitted_F'
params_name = 'w_GC/w_GC_0'
stim_name = 'smooth_0.81'
filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}'


with open(f'{filepath}/params', 'rb') as handle:
    params = pickle.load(handle)



connecter = connectivity(params,
                        filepath = filepath)

W_BB = connecter.weight_matrix_i_to_i(-1/params['tauB'],params['nb_cells'])
W_BA = connecter.weight_matrix_i_to_nn(-1*params['wBA'],params['nb_cells'])

W_AA = connecter.weight_matrix_i_to_i(-1/params['tauA'],params['nb_cells'])
W_AB = connecter.weight_matrix_i_to_nn(params['wAB'],params['nb_cells'])



W_connectivity_B = (W_BB,W_BA) 
W_connectivity_A = (W_AB,W_AA)
L = connecter.assemble_matrix_IPL([W_connectivity_B,W_connectivity_A])

[lam,P] = np.linalg.eig(L)
P_inv = np.linalg.inv(P)

print(lam)



# 
P@np.exp(lam*t)@P_inv