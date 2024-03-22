from stimuli import stim_moving_object_for_2D_net
from connectivity import connectivity
from system import system
from plotting import plotting
from nonlinearities import N
from utils  import GainF_B,GainF_G, DOG
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import json
import sys

SPEED = 3.0
W = 0
save = False
filepath = sys.argv[1]
print(filepath)
# net_name = f'bipolar_pooling_lateral_randpos'
# stim_type = 'smooth'
# param = 'wAB'
# params_name = f'{param}/{param}_{60}'
# stim_name = f'{stim_type}_{4.0}'
# filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}'
# filepath = f'{filepath}/{params_name}/{stim_name}'


with open(f'{filepath}/params', 'rb') as handle:
    params = pickle.load(handle)

 
print('simulation runs')


# create stimulus
stimulus_maker = stim_moving_object_for_2D_net(params,
                                                filepath = filepath)
# inp = stimulus_maker.smooth_motion()

bar = stimulus_maker.bar_smooth()
#tkern = stimulus_maker.alpha_kernel()

tkern = stimulus_maker.load_filter()
#tkern = stimulus_maker.filter_biphasic_norm()
# plt.plot(tkern)
# plt.show()
_,inp = stimulus_maker.OPL()
inp = inp*400#*(1/params['tauB'])

params = stimulus_maker.add_params()
stimulus_maker.plot_stim()
stimulus_maker.plot_kernels()
params = stimulus_maker.add_params()

# create weight matrices

connecter = connectivity(params,
                        filepath = filepath)

W_BB = connecter.weight_matrix_i_to_i(-1/params['tauB'],params['nb_cells'])
W_BA = connecter.weight_matrix_i_to_nn(params['wBA'],params['nb_cells'])

W_AA = connecter.weight_matrix_i_to_i(-1/params['tauA'],params['nb_cells'])
W_AB = connecter.weight_matrix_i_to_i(params['wAB'],params['nb_cells'])


W_GG = connecter.weight_matrix_i_to_i(-1/params['tauG'],params['nb_GC_cells'])
W_outB = connecter.weight_matrix_pooling(params['wGB'])
W_outA = connecter.weight_matrix_pooling(params['wGA'])


W_ActB = connecter.weight_matrix_i_to_i(-1/params['tauActB'],params['nb_cells'])
W_BtoActB = connecter.weight_matrix_i_to_i(params['hB'],params['nb_cells'])

W_ActA = connecter.weight_matrix_i_to_i(-1/params['tauActA'],params['nb_cells'])
W_AtoActA = connecter.weight_matrix_i_to_i(params['hA'],params['nb_cells'])

W_ActG = connecter.weight_matrix_i_to_i(-1/params['tauActG'],params['nb_GC_cells'])
W_GtoActG = connecter.weight_matrix_i_to_i(params['hG'],params['nb_GC_cells'])

W_krecB = connecter.weight_matrix_i_to_i(params['krecB'],params['nb_cells'])
W_krelB = connecter.weight_matrix_i_to_i(params['krelB']*params['betaB'],params['nb_cells'])

W_krecA = connecter.weight_matrix_i_to_i(params['krecA'],params['nb_cells'])
W_krelA = connecter.weight_matrix_i_to_i(params['krelA']*params['betaA'],params['nb_cells'])



W_connectivity_B = (W_BB,W_BA) 
W_connectivity_A = (W_AB,W_AA)
connecter.assemble_matrix_IPL([W_connectivity_B,W_connectivity_A])
connecter.plot_weight_matrix_IPL()
connecter.plot_weight_matrix_pooling(W_outB)

params = connecter.get_eig()
params = connecter.add_params()

# create and solve the system
sys = system(params, W_GG, W_ActG, W_GtoActG)

sys.create_layer([*W_connectivity_B],
                W_ActB,W_BtoActB,
                W_krecB,W_krelB,
                W_outB,
                params['rectification_BC'],
                inp)


sys.create_layer([*W_connectivity_A],
                W_ActA,W_AtoActA,
                W_krecA,W_krelA,
                W_outA,
                params['rectification_AC'],
                np.zeros(inp.shape))


#sys.dummy()
test,test2,test3 = sys.solve_IPL_GainControl_Plasticity(GainF_B,N)
Layers = sys.Layers_IPL
#res,A = sys.solve_IPL_GainControl(N)

VGsys,AGsys,NGsys = sys.solve_GC(N)
RGsys, GGsys = sys.rectify(N,GainF_G)
PVA = sys.PVA()


nb_cells = params['nb_cells']
tps = params['tps']

VB = np.zeros((nb_cells,tps))
NB = np.zeros((nb_cells,tps))
AB = np.zeros((nb_cells,tps))
GB = np.zeros((nb_cells,tps))
RB = np.zeros((nb_cells,tps))

for c in range(nb_cells):

    VB[c,:] = Layers[0]['X'][c]
    #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
    NB[c,:] = Layers[0]['X_rect'][c]
    AB[c,:] =  Layers[0]['A'][c]
    GB[c,:] = Layers[0]['G'][c] #[GainF_B(a) for a in AB[c,:]]
    RB[c,:] = NB[c,:]*GB[c,:]
    
    
VA = np.zeros((nb_cells,tps))
NA = np.zeros((nb_cells,tps))
AA = np.zeros((nb_cells,tps))
GA = np.zeros((nb_cells,tps))
RA = np.zeros((nb_cells,tps))

for c in range(nb_cells):

    VA[c,:] = Layers[1]['X'][c]
    #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
    NA[c,:] = Layers[1]['X_rect'][c]
    AA[c,:] =  Layers[1]['A'][c]
    GA[c,:] = Layers[1]['G'][c] #[GainF_B(a) for a in AB[c,:]]
    RA[c,:] = NA[c,:]*GA[c,:]

[ant_time,ant_space] = sys.calculate_anticipation()



VG = np.zeros((nb_cells,tps))
NG = np.zeros((nb_cells,tps))
AG = np.zeros((nb_cells,tps))
GG = np.zeros((nb_cells,tps))
RG = np.zeros((nb_cells,tps))

for c in range(nb_cells):

    VG[c,:] =VGsys[c]
    NG[c,:] = NGsys[c]#[N(v,params,'GC')for v in VG[c,:]]
    AG[c,:] =  AGsys[c]
    GG[c,:] = GGsys[c]#[GainF_G(a) for a in AG[c,:]]
    RG[c,:] = NG[c,:]*GG[c,:]


params['ant_time'] =ant_time
params['ant_space'] =ant_space

out = {'res' : Layers,
    'VB': VB,
    'AB' : AB,
    'NB' : NB,
    'GB' : GB,
    'RB' : RB,
    'VA': VA,
    'AA' : AA,
    'NA' : NA,
    'GA' : GA,
    'RA' : RA,
    'VG' : VG,
    'AG': AG,
    'GG': GG,
    'NG': NG,
    'RG' : RG,
    'PVA': PVA,
    'inp': inp
        }


# plot
plotter = plotting(params,out,filepath= filepath)
plotter.plot_all_BC_responses(layer = 0, response = 'RB')
plotter.plot_all_AC_responses(layer = 1)
plotter.plot_all_GC_responses(title = f'Pooled response, Anticipation {np.round(ant_time.mean(),3)} s, {np.round(ant_space.mean(),3)} mm')
# plt.figure()
# plt.plot(test)
# plt.figure()
# plt.plot(test2)
# plt.figure()
# plt.plot(test3)
# plt.show()
# save whole simulation 
if save == True:
    print('saving')
    with open(f'{filepath}/out', 'wb') as handle:
        pickle.dump(out, handle)
        
    with open(f'{filepath}/params', 'wb') as handle:
        pickle.dump(params, handle)


# only save maximum
# with open(f'{filepath}/params.json', 'wb') as handle:
#     json.dump(params, handle)
