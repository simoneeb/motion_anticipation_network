import os
import numpy as np
from joblib import Parallel
from run_Reciporcal_plusA2 import run_Reciporcal_plusA2
from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params
import time

#net_name = f'Reciporcal_mono_linear_heavy_maxan_equalweight/noGCGainControl'
net_name = f'Reciporcal_mono_linear_plastic_opt/noGCGainControl/fixed'
#net_name = f'Reciporcal_mono_linear_plastic_opt/noGCGainControl/'

stim_type = 'smooth'

#loop over parameter
param = 'wBA'
vals =[0.0,46.0]

speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,0.27,0.81,1.62,3.24]
#speeds = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,0.27,0.81,1.62,3.24]
#speeds = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,0.27,0.81,1.62,3.24]
#speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]
#speeds = [0.1,0.4,0.7,1.0,2.0]
#speeds = [0.2]

start = time.time()


for val in vals:
    val = np.round(val,2)
    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')
    # loop over speeds : 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/{net_name}'
        print(f'speed = {si}')
        params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
        ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  
        os.system(f'python plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {val} {stim_name}')
    
    # os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val} ')
    # os.system(f'python plot_speeds_auto.py {filepath} {stim_type} {param} {val} ')
    # #os.system(f'python plot_pva.py {filepath} {stim_type} {param} {val} ')
    # os.system(f'python plot_speeds_gaincontrol_mechanism.py {filepath} {stim_type} {param} {val}')
    # #os.system(f'python plot_speeds_lateral_mechanism.py {filepath} {stim_type} {param} {val}')

stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))