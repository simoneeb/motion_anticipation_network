import json
import pickle
import os
import numpy as np



def make_params(param_names = None, param_vals = None, filepath = None):


    # define parameter
    nb_cells = 512     # numner of BC and AC cells
    nb_GC_cells = 512   # number of GC cells, set same as nb_cells
    saving_range = 50    # numner of BC responses that are saved when only one GC is saved
    std_GC = 0.065       # sigma for pooling GC center
    std_GC_s = 0.485     # sigma for pooling GC surround,  not used

    
    rf_BC = 0.05*6       # sigma for spatial convolution ~ BC receprive field center size 
    rf_BC_s = 0.20*6     # sigma for spatial convolution ~ BC receprive field surroud size 

   

    spacing = 0.005     # delta  for distance between cells [mm]
    dt = 0.001          # integration time tep [s]


    speed = 0.81              # bar speed [mm/s]
    bar_width =0.160          # half bar width [mm]
    stimulus_polarity = 1     # polarity of the bar, 1 for white and -1 for black 
    stop = None               # if value [mm], bar stops moving at this position 
    start_cell = 150          # if motion onset, bar starts moving at this position
    start_tp = 1              # i don't rember       
    occluder_width = 2*std_GC # if bar dissapears behind occluder, widhth of the occluder 

    w_BC = 0.     # surround weight of spatial convolution, BC receptive field
    w_GC = 0.0    # surround weight for GC pooling, if 0 no surround 

    tauA = 0.15   # Amacrine time constant 
    tauB = 0.08   # Bipolar time constant
    tauG = 0.01   # Ganglion Time constant

    tauOPL = 0.04      # convolution time constant
    tauOPL2 = 0.0876   # if biphasic, rebound time constanst 
    SF =0.             # monphasic if 0, biphasic is 1

    input_scale = 20.   # scale factor for input amplitude TODO set to mV to have realistic amplitudes at the consecutive levels

    wAB = 10.          #  weight from bipolar to amacrine
    wBA = 10.           #  weight from amacrine to bipolar

  
    d = 1              # if connectivity is not nearest neightbors, distance for cennections
    wGB = 4.#0.0400       # weight of BC-GC pooling
    wGA = 0.0          # weight of AC-GC pooling
    wGA2 = 0.0

    rectification_BC = False   # if True, BC outputs are rectified
    rectification_AC = False   # if True, AC outputs are rectified
    rectification_n = True     # has to be True, such that voltage that guides occupancy is rectified

    slope_BC = 1               # slope of BC/AC rectifiaction in rectification_BC is True
    threshold_BC = 0           # thresold of BC/AC rectifiaction in rectification_BC is True


    slope_GC = 30           # slope of GC rectifiaction for transformation voltage to firing rate
    threshold_GC = 0.0         # threshold of GC rectifiaction for transformation voltage to firing rate

    slope_n = 1                # slope of rectifiaction for occupancy input
    threshold_n = 0.0          # threshold of rectifiaction for occupancy input

    krecB = 1.  # recovery rate plasticity in B
    krelB = .5  # release rate plasticiy in B
    betaB = 0.0 # input scale factir plasicity in B

    krecA = 1. # recovery rate plasticity in B
    krelA = .5 # release rate plasticiy in B
    betaA = 0. # input scale factir plasicity in B

    X0 = 0   # intital condition

    tauActB = 0.12  # activity time constant for gain control in bipolars
    hB = 0. # gain control strength in bipolars

    tauActA = 0.1  # activity time constant for gain control in amacrines
    hA = 0.0 # gain control strength in amacrines

    tauActG = 0.1995  # activity time constant for gain control in GCs
    hG =0.            # gain control strength in bipolars

    # make dict
    params = { 'nb_cells' : nb_cells,
                'nb_GC_cells' : nb_GC_cells,
                'saving_range' : saving_range,
                'rf_BC' : rf_BC,
                'rf_BC_s' : rf_BC_s,

                'std_GC' : std_GC,
                'std_GC_s' : std_GC_s,
                'spacing' : spacing,
                'dt' : dt,

                'speed' : speed,
                'bar_width' : bar_width,
                'stimulus_polarity' : stimulus_polarity,

                'start_cell': start_cell,
                'start_tp': start_tp,
                'occluder_width': occluder_width,

                'stop_pos' : stop,
                'w_BC' : w_BC,
                'w_GC' : w_GC,

                'X0': X0,
                'tauA' : tauA, 
                'tauB' : tauB,
                'tauG' : tauG,
                'tauOPL' : tauOPL,
                'tauOPL2' : tauOPL2,
                'SF' : SF,
                'input_scale' : input_scale,

                'wAB' : wAB,
                'wBA' : wBA, 

                'wGB' : wGB, 
                'wGA' : wGA, 
                'wGA2' : wGA2, 
                'd' : d,

                'rectification_BC' :rectification_BC,
                'slope_BC' : slope_BC,
                'threshold_BC' : threshold_BC,

                'rectification_AC' : rectification_AC,
                'slope_GC' : slope_GC,
                'threshold_GC' : threshold_GC,

                'rectification_n' : rectification_n,
                'slope_n' : slope_n,
                'threshold_n' : threshold_n,

                'krecB' : krecB,
                'krelB' : krelB,
                'betaB' : betaB,

                'krecA' : krecA,
                'krelA' : krelA,
                'betaA' : betaA,

                'tauActB' : tauActB,
                'hB' : hB,
                            
                'tauActA' : tauActA,
                'hA' : hA,

                'tauActG' : tauActG,
                'hG' : hG
    }

    
    if param_names is not None:
        for i,nam in enumerate(param_names):
            params[nam] = param_vals[i]


    distance = nb_cells*spacing 
    duration = distance/params['speed']
    time = np.arange(0,duration,dt)
    tps = len(time)

    pos_rf_mid = np.linspace(0,distance,nb_cells+2) #mm
    pos_rf_mid = pos_rf_mid[1:-1] #mm
    cell_spacing = np.mean(np.diff(pos_rf_mid))
    rf_overlap = rf_BC/cell_spacing


    params['distance'] = distance
    params['duration'] = duration
    params['rf_overlap'] = rf_overlap
    params['cell_spacing'] = cell_spacing
    params['tps'] = tps


    if filepath is not None:

        if not os.path.isdir(filepath):
            os.makedirs(filepath)

        if not os.path.isdir(f'{filepath}/plots'):
            os.makedirs(f'{filepath}/plots')

        with open(f'{filepath}/params', 'wb') as handle:
            pickle.dump(params, handle)

        with open(f'{filepath}/params.json', 'w', encoding='utf-8') as handle:
            json.dump(params, handle,indent=4)

    return params
   

def modify_params(params, param_names,values):
    for i,parami in enumerate(param_names):
        params[parami] = values[i]

    nb_cells = params['nb_cells']
    dt = params['dt']
    spacing = params['spacing']
    rf_BC = params['rf_BC']

    distance = nb_cells*spacing  
    duration = distance/params['speed']
    time = np.arange(0,duration,dt)
    tps = len(time)


    pos_rf_mid = np.linspace(0,distance,nb_cells) #mm
    #pos_rf_mid = pos_rf_mid[1:-1] #mm
    cell_spacing = np.mean(np.diff(pos_rf_mid))
    rf_overlap = rf_BC/cell_spacing


    params['distance'] = distance
    params['duration'] = duration
    params['rf_overlap'] = rf_overlap
    params['cell_spacing'] = cell_spacing
    params['tps'] = tps



    return params



def load_params(filepath,params_name):

    with open(f'{filepath}/{params_name}', 'rb') as handle:
        params = pickle.load(handle)
        
    return params
