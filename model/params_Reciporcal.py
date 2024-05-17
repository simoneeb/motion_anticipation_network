import json
import pickle
import os
import sys
import os
import numpy as np



#filepath = sys.argv[1]
def make_params(param_names = None, param_vals = None, filepath = None):


    # define parameter
    nb_cells = 450
    nb_GC_cells = 450
    saving_range = 50
    rf_GC = 0.065*6 # 0.09s
    rf_GC_s = 0.485*6 # 0.09s
    rf_BC = 0.05*6 # 0.09s
    rf_BC_s = 0.20*6 # 0.09s

    std_GC = rf_GC/6
    std_GC_s = rf_GC_s/6

    spacing = 0.005  
    dt = 0.01#1/40# 0.001


    speed = 0.81 
    bar_width = 0.150 
    stimulus_polarity = 1
    stop = None
    start_cell = 150#-16
    start_tp = 1 #s
    occluder_width = 2*std_GC#s

    #stop = 2 #mm
    w_BC = 0.4 # 0.5
    w_GC = 0.0 # 0.5

    tauA = 0.28#0.156 #0.218 #= RAM mono linear fitted to ACM   # RAM fitted to ACM  = 0.156  
    tauA2 = 0.3   #0.017
    tauB =0.008   #0.01
    tauG = 0.01   # 0.01  #0.01

    tauOPL = 0.04 #0.086   #0.05508089
    tauOPL2 = 0.0876 #0.05730816
    SF =0.# 1.
    input_scale =0.1#0.1 #5.#0.1#0.025#5# 800#459#800

    wAB = 46.#22.   # 10.#44. #= RAM mono linear fitted to ACM   # RAM fitted to ACM = 22
    wBA = 46.# 31.#31. #= RAM mono linear fitted to ACM   # RAM fitted to ACM = 31

    wA2B = 0.#46.
    wA2A = 0. #46.0
    wAA2 = 0.
    wBA2 = 0.#16.#46.

    d = 1
    wGB = .0400
    wGA = -0.004
    wGA2 = 0.0

    rectification_BC = True
    rectification_AC = True
    rectification_n = True

    slope_BC = 1
    threshold_BC = 0 #7.35


    slope_GC = 1110
    threshold_GC = 0.0

    slope_n = 1
    threshold_n = 0.0

    plastic_to_G = False
    plastic_to_A = False
    krecB = 1.  #0.78# 2.0
    krelB = .5  #0.64# 2.0
    betaB =0.0  #0.6 #0.5#0.1  #0.3#.1#3#0.1#.30 #1360.0


    krecA = 1. # 3#0.066#2.#35.66#2.0
    krelA = .5 # 6#0.0015#2.#0.54
    betaA =0.0 # 0.03#.0.04 #0.2#0.5# 0.03#0.3#2#1#.3#0.1#.300 #1360.0


    krecA2 = .1 #3#0.066#2.#35.66#2.0
    krelA2 = .1 #6#0.0015#2.#0.54
    betaA2 =0.0 #0.6#0.2#0.5# 0.03#0.3#2#1#.3#0.1#.300 #1360.0

    X0 = 0

    tauActB = 0.12
    hB = 0. #0.00051#0.511  #611.0#

    tauActA = 0.1
    hA = 0.0 #10110.0 #10000.1

    tauActG = 0.1995
    hG =0.# 0.459 #0.0659


    params = { 'nb_cells' : nb_cells,
                'nb_GC_cells' : nb_GC_cells,
                'saving_range' : saving_range,
                'rf_BC' : rf_BC,
                'rf_BC_s' : rf_BC_s,
                'rf_GC' : rf_GC,
                'rf_GC_s' : rf_GC_s,

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
                'tauA2' : tauA2, 
                'tauB' : tauB,
                'tauG' : tauG,
                'tauOPL' : tauOPL,
                'tauOPL2' : tauOPL2,
                'SF' : SF,
                'input_scale' : input_scale,

                'wAB' : wAB,
                'wBA' : wBA, 

                'wA2B' : wA2B, 
                'wA2A' : wA2A, 
                'wAA2' : wAA2, 
                'wBA2' : wBA2, 

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

                'plastic_to_G' : plastic_to_G,
                'plastic_to_A' : plastic_to_A,
                'krecB' : krecB,
                'krelB' : krelB,
                'betaB' : betaB,

                'krecA' : krecA,
                'krelA' : krelA,
                'betaA' : betaA,


                'krecA2' : krecA2,
                'krelA2' : krelA2,
                'betaA2' : betaA2,

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


    distance = nb_cells*spacing #599*0.005 # from spacing between cells 30 mum
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
