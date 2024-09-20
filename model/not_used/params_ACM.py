import json
import pickle
import os
import sys
import os
import numpy as np



# filepath = sys.argv[1]

# if not os.path.isdir(filepath):
#     os.makedirs(filepath)
# if not os.path.isdir(f'{filepath}/plots'):
#     os.makedirs(f'{filepath}/plots')


def make_params(param_names = None, param_vals = None, filepath = None):

    # define parameter

    nb_cells = 600#600
    nb_GC_cells = 600#300

    #rf_GC = 0.045*6#0.085*6 # 0.09s
    rf_GC = 0.085*6#0.085*6 # 0.09s
    rf_GC = 0.065*6#0.085*6 # 0.09s
    rf_GC_s = 0.485*6 # 0.09s
    rf_BC = 0.05*6 # 0.09s
    rf_BC_s = 0.20*6 # 0.09s


    std_GC = rf_GC/6
    std_GC_s = rf_GC_s/6
    #rf_overlap = 3*6

    spacing = 0.005  
    distance = nb_cells*spacing #599*0.005 # from spacing between cells 30 mum
    dt = 0.001

    speed = 0.81 
    bar_width = 0.080 
    stimulus_polarity = 1
    stop = None
    start_cell = 300#-16
    start_tp = 1 #s
    occluder_width = std_GC #s
    


    #stop = 2 #mm
    w_BC = 0.4 # 0.5
    w_GC = 0.0 # 0.5

    tauA = 0.1 #0.017
    tauA2 = 0.3 #0.017
    tauB = 0.01
    tauG = 0.01

    #0.03825167 0.1199695  0.37569818
    tauOPL = 0.04 #0.086 #0.05508089
    tauOPL2 = 0.0876 #0.05730816
    SF = 1.
    input_scale = .16#0.05#5

    wAB = 0
    wBA = 0
    wBA2 = 0

    d = 1
    wGB = 0.0005#.00030
    wGA = 0.0
    wGA2 = 0.0

    rectification_BC = True
    rectification_AC = False

    slope_BC = 1
    threshold_BC = 7.35


    slope_GC = 1110
    threshold_GC = 0.0

    krecB = 20.0
    krelB = 20.0
    betaB = 0# 300.0 #1360.0


    krecA = 20.0
    krelA = 19.0
    betaA = 0#240.0 #1360.0

    X0 = 0

    tauActB = 0.1
    hB = 0.00611  #611.0#

    tauActA = 0.1
    hA = 0.0 #10110.0 #10000.1

    tauActG = 0.1995
    hG = 0.000459#0.0659





    params = { 'nb_cells' : nb_cells,
                'nb_GC_cells' : nb_GC_cells,
                'rf_BC' : rf_BC,
                'rf_BC_s' : rf_BC_s,
                'rf_GC' : rf_GC,
                'rf_GC_s' : rf_GC_s,

                'std_GC' : std_GC,
                'std_GC_s' : std_GC_s,

                'speed' : speed,
                'bar_width' : bar_width,
                'stimulus_polarity' : stimulus_polarity,

                'start_cell': start_cell,
                'start_tp': start_tp,
                'occluder_width': occluder_width,

                'spacing' : spacing,
                'distance' : distance,
                'dt' : dt,
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

    # if len(sys.argv) > 3:
    #     params[f'{sys.argv[2]}'] = float(sys.argv[3])
        
        
    # if len(sys.argv) > 5:
    #     params[f'{sys.argv[4]}'] = float(sys.argv[5])


    # if len(sys.argv) > 7:
    #     params[f'{sys.argv[6]}'] = float(sys.argv[7])

    if param_names is not None:
        for i,nam in enumerate(param_names):
            #print(nam,param_vals[i])
            params[nam] = param_vals[i]

    duration = distance/params['speed']
    time = np.arange(0,duration,dt)
    tps = len(time)


    pos_rf_mid = np.linspace(0,distance,nb_cells+2) #mm
    pos_rf_mid = pos_rf_mid[1:-1] #mm
    cell_spacing = np.mean(np.diff(pos_rf_mid))
    rf_overlap = rf_BC/cell_spacing


    params['duration'] = duration
    params['rf_overlap'] = rf_overlap
    params['cell_spacing'] = cell_spacing
    params['tps'] = tps



    # with open(f'{filepath}/params', 'wb') as handle:
    #     pickle.dump(params, handle)


    # # filepath = '/user/sebert/home/Documents/Simulations/motion/anticipation_1D/new/bipolar_pooling_lateral_OFF_laplacian_GainControl/betaA/betaA_0/smooth_2.6'
    # # with open(f'{filepath}/params', 'rb') as handle:
    # #     params = pickle.load(handle)


    # # print(params)
    # with open(f'{filepath}/params.json', 'w', encoding='utf-8') as handle:
    #     json.dump(params, handle,indent=4)



    # print(params)
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

    # with open(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/bipolar_pooling_lateral/w/w_60/smooth_4.0/params', 'rb') as handle:
    #     params =  pickle.load(handle)