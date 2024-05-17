import os
import numpy as np
import time

from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params, load_params,modify_params


'''
script to loop over values for one parameter and simulate  model respons eto different speeds
'''


net_name = f'reciprocal_ff_fitted_cell_431'
stim_type = 'smooth'

param = 'wBA'       # parameter to loop over
vals =[10.,20.] #[46.0]        # values to test 

speeds = [0.14,0.42,0.7,0.98,1.96]

start = time.time()

for val in vals:
    val = np.round(val,3)
    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')
    # loop over speeds : 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        #filepath = f'/Users/simone/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        filepath_p = f'/Users/simoneebert/Documents/Simulations/motion_anticipation_network/{net_name}'
        filepath = f'/Users/simoneebert/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        print(f'speed = {si}')
        #params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
     
        params = load_params(filepath_p,'params')
        params = modify_params(params, param_names= ['speed',param,'dt'], values=[si,val,0.001])
        ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  
        os.system(f'python plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {val} {stim_name}')

    os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val}')
    
stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))