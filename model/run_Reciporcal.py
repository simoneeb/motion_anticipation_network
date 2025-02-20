from stimuli import stim_moving_object_for_2D_net
from connectivity import connectivity
from system import system
from nonlinearities import N
from utils  import GainF_B,GainF_G, DOG,measure_onset_anticipation
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle 



def run_Reciporcal(params, filepath = None, save_one = True, measure_n = False, stim_type = 'smooth',step_stop = None):
    
   
   
    """
    runs the simulation of a reciprocal amacrine network for a given parameterset

    params (dict) : parameter of the model
    filepath (string): directory to store the output in 
    save_one : if True, the reponse of only the middle cell of the network is saved
    measure_n : if True, occulancy min is returned
    stim_type (string) : stimulus to use for the simulation
    step_stop : if stim_type is 'step', the time st which the step stimulus ends

    returns: 

    max_RG:
    max_RB:
    max_drive:
    params['tps_rf_GC_mid']:
    [middle_cell_GC]:
    onset_RG:
    onset_RB:
    RG[middle_cell_GC,:]:
    RB[middle_cell_GC,:]:
    VG[middle_cell_GC,:]:




    """


    # create directory to save outputs
    if filepath is not None:

        if not os.path.isdir(filepath):
            os.makedirs(filepath)
        if not os.path.isdir(f'{filepath}/plots'):
            os.makedirs(f'{filepath}/plots')


    

    # create stimulus
    stimulus_maker = stim_moving_object_for_2D_net(params,
                                                    filepath = filepath)


    if stim_type == 'smooth':
        bar = stimulus_maker.bar_smooth()

    if stim_type == 'onset':
        bar = stimulus_maker.bar_onset()

    if stim_type == 'reversing':
        bar = stimulus_maker.bar_reversing()

    if stim_type == 'interrupted':
        bar = stimulus_maker.bar_interrupted()

    if stim_type == 'impulse':
        bar = stimulus_maker.impulse_stimulus()

    if stim_type == 'step':
        if step_stop is not None:
            bar = stimulus_maker.step_stimulus(stop = step_stop)
        else:
            bar = stimulus_maker.step_stimulus()

    #_ = stimulus_maker.load_filter()                      # load filter from data and fit convolution params to it 
    tkern = stimulus_maker.filter_biphasic_norm()          # make filter based on parameter

    spat,inp = stimulus_maker.OPL()                        # simulate OPS response
    F_inp = stimulus_maker.F()                             # Simulate in put into dynamical system
 
    if filepath is not None:
        stimulus_maker.plot_stim()                      
        stimulus_maker.plot_kernels()


    params = stimulus_maker.add_params()                # calculate additional params

    # create weight matrices
    connecter = connectivity(params,
                            filepath = filepath)


    # connectivity matrices
    W_BB = connecter.weight_matrix_i_to_i(-1/params['tauB'],params['nb_cells'])         # timeconstants B
    W_BA = connecter.weight_matrix_i_to_nn(-1*params['wBA'],params['nb_cells'])         # inputs from amacrines

    W_AA = connecter.weight_matrix_i_to_i(-1/params['tauA'],params['nb_cells'])         # timeconstants A
    W_AB = connecter.weight_matrix_i_to_nn(params['wAB'],params['nb_cells'])            # inputs from bipolars

    # create pooling matrices
    W_GG = connecter.weight_matrix_i_to_i(-1/params['tauG'],params['nb_GC_cells'])      # timeconstants G
    W_outB = connecter.weight_matrix_pooling(params['wGB'])                             # pooling over B
    W_outA = connecter.weight_matrix_pooling(-1*params['wGA'])                          # pooling over A

    # matrices for gain control
    W_ActB = connecter.weight_matrix_i_to_i(-1/params['tauActB'],params['nb_cells'])    # time constant of activity decrease of for Gain Control in B
    W_BtoActB = connecter.weight_matrix_i_to_i(params['hB'],params['nb_cells'])         # stength of gain control

    W_ActA = connecter.weight_matrix_i_to_i(-1/params['tauActA'],params['nb_cells'])    # time constant of activity decrease of for Gain Control in A
    W_AtoActA = connecter.weight_matrix_i_to_i(params['hA'],params['nb_cells'])         # stength of gain control

    W_ActG = connecter.weight_matrix_i_to_i(-1/params['tauActG'],params['nb_GC_cells'])  # time constant of activity decrease of for Gain Control in G
    W_GtoActG = connecter.weight_matrix_i_to_i(params['hG'],params['nb_GC_cells'])       # stength of gain control


    # matrices for plasticity
    W_krecB = connecter.weight_matrix_i_to_i(params['krecB'],params['nb_cells'])         # recovery for each BC
    W_krelB = connecter.weight_matrix_i_to_i(params['krelB']*params['betaB'],params['nb_cells'])   # relase for each BC

    W_krecA = connecter.weight_matrix_i_to_i(params['krecA'],params['nb_cells'])                    # recovery for each AC
    W_krelA = connecter.weight_matrix_i_to_i(params['krelA']*params['betaA'],params['nb_cells'])    # release for each AC


    # collect matrices for B and A
    W_connectivity_B = (W_BB,W_BA) 
    W_connectivity_A = (W_AB,W_AA)
    connecter.assemble_matrix_IPL([W_connectivity_B,W_connectivity_A]) # makes ones big connectivity matrix for the IPL

    # plot connectivity
    if filepath is not None:
        L = connecter.plot_weight_matrix_IPL()
        connecter.plot_weight_matrix_pooling(W_outB)


    params = connecter.get_eig()      # calculate eigenvalued of the connectivitymatrix
    params = connecter.add_params()   # calculate more parameter of the connectivty 

    # create the system
    sys = system(params, W_GG, W_ActG, W_GtoActG)


    # create bipolar layer
    sys.create_layer([*W_connectivity_B],
                    W_ActB,W_BtoActB,
                    W_krecB,W_krelB,
                    W_outB,
                    params['rectification_BC'],
                    F_inp)

    # creade amacrine layer
    sys.create_layer([*W_connectivity_A],
                    W_ActA,W_AtoActA,
                    W_krecA,W_krelA,
                    W_outA,
                    params['rectification_AC'],
                    np.zeros(inp.shape))



    # solve the system 
    sys.solve_IPL_GainControl_Plasticity(GainF_B,N)
    Layers = sys.Layers_IPL

    #res,A = sys.solve_IPL_GainControl(N)

    # solve GC layer and rectify output 
    VGsys,AGsys,NGsys = sys.solve_GC(N)
    RGsys, GGsys = sys.rectify(N,GainF_G)
    PVA = sys.PVA()


    nb_cells = params['nb_cells']
    tps = params['tps']


    # collect the output bipolar 
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
        
    # collect output ganglion
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

    [ant_time,ant_space,ant_time_drive,ant_space_drive] = sys.calculate_anticipation()


    # colllect output GC
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

    # crop the output ans save
    middle_cell_BC = int(params['nb_cells']/2)
    middle_cell_GC = int(params['nb_GC_cells']/2)
    ran = params['saving_range']
    if save_one is True:
        out = {
            # 'res' : Layers,
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
            'VG' : VG[middle_cell_GC,:],
            'AG': AG[middle_cell_GC,:],
            'GG': GG[middle_cell_GC,:],
            'NG': NG[middle_cell_GC,:],
            'RG' : RG[middle_cell_GC,:],
            # 'PVA': PVA,
            'inp': inp[middle_cell_GC,:],
            'spat': spat[middle_cell_GC,:],
            'F':F_inp[middle_cell_GC,:],
                }
    if save_one is False:
        out = {
            # 'res' : Layers,
            'VB': VB[:,:],
            'RB' : RB[:,:],
            'VA': VA[:,:],
            
            'RA' : RA[:,:],
            'VG' : VG[:,:],
          
            'RG' : RG[:,:],
            # 'PVA': PVA,
            'inp': inp[:,:],
            'spat': spat[:,:],
            'F':F_inp[:,:],
                }
             
  
    dt = params['dt']
    max_RG = np.argmax(RG[middle_cell_GC,:])*dt
    max_RB = np.argmax(RB[middle_cell_GC,:])*dt
    max_drive = np.argmax(F_inp[middle_cell_GC,:])*dt

    max_amp_B = np.max(VB[middle_cell_GC,:])

    if filepath is not None:

        print('saving output')
        with open(f'{filepath}/out_{stim_type}', 'wb') as handle:
            pickle.dump(out, handle)
            
        with open(f'{filepath}/params', 'wb') as handle:
            pickle.dump(params, handle)


    # measure onset anticipation, not used anymore 
    onset_RG = measure_onset_anticipation( RG[middle_cell_GC,:])
    onset_RB = measure_onset_anticipation( RB[middle_cell_GC,:])

    # get the minimal occpancy value, not used anymore
    nmin_B = np.min(OB[middle_cell_BC,:-1])
    nmin_A = np.min(OA[middle_cell_BC,:-1])

    
    if measure_n is True:
        return [max_RG,max_RB,max_drive,params['tps_rf_GC_mid'][middle_cell_GC], onset_RG,onset_RB,RG[middle_cell_GC,:],RB[middle_cell_GC,:],nmin_B,nmin_A]
    else:
        return [max_RG,max_RB,max_drive,params['tps_rf_GC_mid'][middle_cell_GC],max_amp_B, onset_RG,onset_RB,RG[middle_cell_GC,:],RB[middle_cell_GC,:],VG[middle_cell_GC,:]]
