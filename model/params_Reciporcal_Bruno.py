import json
import pickle
import os
import sys
import os
import numpy as np



filepath = sys.argv[1]

if not os.path.isdir(filepath):
    os.makedirs(filepath)
if not os.path.isdir(f'{filepath}/plots'):
    os.makedirs(f'{filepath}/plots')



# define parameter

nb_cells = 60
nb_GC_cells = 60
saving_range = 10
rf_GC = 0.15*6 # 0.09s
rf_GC_s = 0.485*6 # 0.09s
rf_BC = 0.06*6 # 0.09s
rf_BC_s = 0.12*6 # 0.09s


std_GC = rf_GC/6
std_GC_s = rf_GC_s/6
#rf_overlap = 3*6
speed = 3.0  
spacing = 0.03 
distance = nb_cells*spacing #599*0.005 # from spacing between cells 30 mum
dt = 0.0001
stimulus_polarity = 1
stop = None
#stop = 2 #mm
w_BC = 0#1.5/2 # 0.5
w_GC = 0.0 # 0.5

tauA = 0.127  #0.09 #0.017
tauA2 = 0.3   #0.017
tauB =0.0034 #0.01
tauG = 0.04 # 0.01  #0.01

#0.03825167 0.1199695  0.37569818
tauOPL = 0.0314 #0.05508089
tauOPL2 = 0.01 #0.05730816
SF = 0.
input_scale =0.25 # 800#459#800

wAB = 10.2 #10.0
wBA = 7.8 #31#-100.0
wBA2 = 0
d = 1
wGB = 10.6
wGA = 3.6
wGA2 = 0.0

rectification_BC = False
rectification_AC = False

slope_BC = 1
threshold_BC = 0


slope_GC = 1110
threshold_GC = 0.0

krecB =0.098#2.#0.78# 2.0
krelB =0.04#2.#0.64# 2.0
betaB =0#.1#3#0.1#.30 #1360.0


krecA = 3#0.066#2.#35.66#2.0
krelA = 6#0.0015#2.#0.54
betaA = 0#2#1#.3#0.1#.300 #1360.0

X0 = 0

tauActB = 0.12
hB = 0 #0.00051#0.511  #611.0#

tauActA = 0.1
hA = 0.0 #10110.0 #10000.1

tauActG = 0.1995
hG = 0#0.459 #0.0659


params = { 'nb_cells' : nb_cells,
            'nb_GC_cells' : nb_GC_cells,
            'saving_range' : saving_range,
            'rf_BC' : rf_BC,
            'rf_BC_s' : rf_BC_s,
            'rf_GC' : rf_GC,
            'rf_GC_s' : rf_GC_s,

            'std_GC' : std_GC,
            'std_GC_s' : std_GC_s,
            'speed' : speed,
            'spacing' : spacing,
            'distance' : distance,
            'dt' : dt,
            'stimulus_polarity' : stimulus_polarity,
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

if len(sys.argv) > 3:
    params[f'{sys.argv[2]}'] = float(sys.argv[3])
    
    
if len(sys.argv) > 5:
    params[f'{sys.argv[4]}'] = float(sys.argv[5])


if len(sys.argv) > 7:
    params[f'{sys.argv[6]}'] = float(sys.argv[7])


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



with open(f'{filepath}/params', 'wb') as handle:
    pickle.dump(params, handle)


# filepath = '/user/sebert/home/Documents/Simulations/motion/anticipation_1D/new/bipolar_pooling_lateral_OFF_laplacian_GainControl/betaA/betaA_0/smooth_2.6'
# with open(f'{filepath}/params', 'rb') as handle:
#     params = pickle.load(handle)


# print(params)
with open(f'{filepath}/params.json', 'w', encoding='utf-8') as handle:
    json.dump(params, handle,indent=4)


# with open(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/bipolar_pooling_lateral/w/w_60/smooth_4.0/params', 'rb') as handle:
#     params =  pickle.load(handle)