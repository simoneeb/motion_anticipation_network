from stimuli import stim_moving_object_for_2D_net
from connectivity import connectivity
from system import system
from plotting import plotting
from nonlinearities import N
from utils  import GainF_B,GainF_G, DOG,measure_onset_anticipation
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import json
import sys

# SPEED = 3.0
# W = 0
# save = False
# save_one = True
# filepath = sys.argv[1]
#print(filepath)
# net_name = f'bipolar_pooling_lateral_randpos'
# stim_type = 'smooth'
# param = 'wAB'
# params_name = f'{param}/{param}_{60}'
# stim_name = f'{stim_type}_{4.0}'
# filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}'
# filepath = f'{filepath}/{params_name}/{stim_name}'




def run_Reciporcal_plusA2(params, filepath = None, save_one = False, measure_n = False):
    


    if filepath is not None:
        with open(f'{filepath}/params', 'rb') as handle:
            params = pickle.load(handle)

    


    # create stimulus
    stimulus_maker = stim_moving_object_for_2D_net(params,
                                                    filepath = filepath)
    # inp = stimulus_maker.smooth_motion()

    bar = stimulus_maker.bar_smooth()
    #tkern = stimulus_maker.alpha_kernel()

    _ = stimulus_maker.load_filter()
    tkern = stimulus_maker.filter_biphasic_norm()
    # plt.plot(tkern)
    # plt.show()
    _,inp = stimulus_maker.OPL()
    F_inp = stimulus_maker.F()
    #inp = inp*params['input_scale']

    if filepath is not None:
        stimulus_maker.plot_stim()
        stimulus_maker.plot_kernels()
    params = stimulus_maker.add_params()

    # create weight matrices

    connecter = connectivity(params,
                            filepath = filepath)

    W_BB = connecter.weight_matrix_i_to_i(-1/params['tauB'],params['nb_cells'])
    W_BA = connecter.weight_matrix_i_to_nn(-1*params['wBA'],params['nb_cells'])
    W_BA2 = connecter.weight_matrix_i_to_nnplusd(params['wBA2'],params['nb_cells'],d = 3)


    W_A2A2 = connecter.weight_matrix_i_to_i(-1/params['tauA2'],params['nb_cells'])
    W_A2B = connecter.weight_matrix_i_to_i(params['wA2B'],params['nb_cells'])
    W_A2A = connecter.weight_matrix_i_to_nn(-1*params['wA2A'],params['nb_cells'])

    W_AA = connecter.weight_matrix_i_to_i(-1/params['tauA'],params['nb_cells'])
    W_AB = connecter.weight_matrix_i_to_nn(params['wAB'],params['nb_cells'])
    W_AA2 = connecter.weight_matrix_i_to_i(-1*params['wAA2'],params['nb_cells'])


    W_GG = connecter.weight_matrix_i_to_i(-1/params['tauG'],params['nb_GC_cells'])
    W_outB = connecter.weight_matrix_pooling(params['wGB'])
    W_outA = connecter.weight_matrix_pooling(params['wGA'])
    W_outA2 = connecter.weight_matrix_pooling(params['wGA2'])


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


    W_krecA2 = connecter.weight_matrix_i_to_i(params['krecA2'],params['nb_cells'])
    W_krelA2 = connecter.weight_matrix_i_to_i(params['krelA2']*params['betaA2'],params['nb_cells'])


    W_connectivity_B = (W_BB,W_BA,W_BA2) 
    W_connectivity_A = (W_AB,W_AA,W_AA2)
    W_connectivity_A2 = (W_A2B,W_A2A,W_A2A2)
    connecter.assemble_matrix_IPL([W_connectivity_B,W_connectivity_A, W_connectivity_A2])
    #L = connecter.plot_weight_matrix_IPL()
    #TODO save L
    #connecter.plot_weight_matrix_pooling(W_outB)

    params = connecter.get_eig()
    params = connecter.add_params()

    # create and solve the system
    sys = system(params, W_GG, W_ActG, W_GtoActG)

    sys.create_layer([*W_connectivity_B],
                    W_ActB,W_BtoActB,
                    W_krecB,W_krelB,
                    W_outB,
                    params['rectification_BC'],
                    F_inp)


    sys.create_layer([*W_connectivity_A],
                    W_ActA,W_AtoActA,
                    W_krecA,W_krelA,
                    W_outA,
                    params['rectification_AC'],
                    np.zeros(inp.shape))
    

    sys.create_layer([*W_connectivity_A2],
                    W_ActA,W_AtoActA,
                    W_krecA2,W_krelA2,
                    W_outA2,
                    params['rectification_AC'],
                    np.zeros(inp.shape))


    #sys.dummy()
    print('simulation runs')
    sys.solve_IPL_GainControl_Plasticity(GainF_B,N)
    Layers = sys.Layers_IPL
    #res,A = sys.solve_IPL_GainControl(N)

    VGsys,AGsys,NGsys = sys.solve_GC(N)
    RGsys, GGsys = sys.rectify(N,GainF_G)
    PVA = sys.PVA()


    nb_cells = params['nb_cells']
    tps = params['tps']

    VB = np.zeros((nb_cells,tps))
    OB = np.zeros((nb_cells,tps))
    NB = np.zeros((nb_cells,tps))
    AB = np.zeros((nb_cells,tps))
    GB = np.zeros((nb_cells,tps))
    RB = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VB[c,:] = Layers[0]['X'][c]
        OB[c,:] = Layers[0]['n'][c]
        #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
        NB[c,:] = Layers[0]['X_rect'][c]
        AB[c,:] =  Layers[0]['A'][c]
        GB[c,:] = Layers[0]['G'][c] #[GainF_B(a) for a in AB[c,:]]
        RB[c,:] = NB[c,:]*GB[c,:]*OB[c,:]
        
        
    VA = np.zeros((nb_cells,tps))
    OA = np.zeros((nb_cells,tps))
    NA = np.zeros((nb_cells,tps))
    AA = np.zeros((nb_cells,tps))
    GA = np.zeros((nb_cells,tps))
    RA = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VA[c,:] = Layers[1]['X'][c]
        OA[c,:] = Layers[1]['n'][c]
        #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
        NA[c,:] = Layers[1]['X_rect'][c]
        AA[c,:] =  Layers[1]['A'][c]
        GA[c,:] = Layers[1]['G'][c] #[GainF_B(a) for a in AB[c,:]]
        RA[c,:] = NA[c,:]*GA[c,:]*OA[c,:]

    VA2 = np.zeros((nb_cells,tps))
    OA2 = np.zeros((nb_cells,tps))
    NA2 = np.zeros((nb_cells,tps))
    AA2 = np.zeros((nb_cells,tps))
    GA2 = np.zeros((nb_cells,tps))
    RA2 = np.zeros((nb_cells,tps))

    for c in range(nb_cells):

        VA2[c,:] = Layers[2]['X'][c]
        OA2[c,:] = Layers[2]['n'][c]
        #NB[c,:] = [N(v,params,'BC')for v in Layers[0]['X'][c]]
        NA2[c,:] = Layers[2]['X_rect'][c]
        AA2[c,:] =  Layers[2]['A'][c]
        GA2[c,:] = Layers[2]['G'][c] #[GainF_B(a) for a in AB[c,:]]
        RA2[c,:] = NA2[c,:]*GA2[c,:]*OA2[c,:]

    [ant_time,ant_space,ant_time_drive,ant_space_drive] = sys.calculate_anticipation()



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
        'OB': OB,
        'AB' : AB,
        'NB' : NB,
        'GB' : GB,
        'RB' : RB,
        'VA': VA,
        'OA': OA,
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
        'inp': inp,
        'F':F_inp,
            }
    # plot

    if filepath is not None:
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

    middle_cell_BC = int(params['nb_cells']/2)
    middle_cell_GC = int(params['nb_GC_cells']/2)
    ran = params['saving_range']
    if save_one:
        out = {'res' : Layers,
            'VB': VB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'OB': OB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'AB' : AB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'NB' : NB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'GB' : GB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'RB' : RB[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'VA': VA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'OA': OA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'AA' : AA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'NA' : NA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'GA' : GA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'RA' : RA[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'VA2': VA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'OA2': OA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'AA2' : AA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'NA2' : NA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'GA2' : GA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'RA2' : RA2[middle_cell_BC-ran:middle_cell_BC+ran,:],
            'VG' : VG[middle_cell_GC,:],
            'AG': AG[middle_cell_GC,:],
            'GG': GG[middle_cell_GC,:],
            'NG': NG[middle_cell_GC,:],
            'RG' : RG[middle_cell_GC,:],
            'PVA': PVA,
            'inp': inp[middle_cell_GC,:],
            'F':F_inp[middle_cell_GC,:],
                }
        
    dt = params['dt']
    max_RG = np.argmax(RG[middle_cell_GC,:])*dt
    max_RB = np.argmax(RB[middle_cell_GC,:])*dt
    max_drive = np.argmax(F_inp[middle_cell_GC,:])*dt

    if filepath is not None:

        print('saving output')
        with open(f'{filepath}/out', 'wb') as handle:
            pickle.dump(out, handle)
            
        with open(f'{filepath}/params', 'wb') as handle:
            pickle.dump(params, handle)

    #return [max_RG,max_RB,max_drive,params['tps_rf_GC_mid'][middle_cell_GC],RG[middle_cell_GC,:],RB[middle_cell_GC,:]]
    onset_RG = measure_onset_anticipation( RG[middle_cell_GC,:])
    onset_RB = measure_onset_anticipation( RB[middle_cell_GC,:])

    nmin_B = np.min(OB[middle_cell_GC,:])
    nmin_A = np.min(OA[middle_cell_GC,:])
    if measure_n is True:
        return [max_RG,max_RB,max_drive,params['tps_rf_GC_mid'][middle_cell_GC], onset_RG,onset_RB,RG[middle_cell_GC,:],RB[middle_cell_GC,:],nmin_B,nmin_A]
    else:
        return [max_RG,max_RB,max_drive,params['tps_rf_GC_mid'][middle_cell_GC], onset_RG,onset_RB,RG[middle_cell_GC,:],RB[middle_cell_GC,:]]



    # only save maximum
    # with open(f'{filepath}/params.json', 'wb') as handle:
    #     json.dump(params, handle)
