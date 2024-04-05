import os
import numpy as np
import time

from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params




'''
script to loop over values for one parameter and simulate  model respons eto different speeds
'''


net_name = f'Reciporcal_test_newsys'
stim_type = 'smooth'

param = 'wBA'       # parameter to loop over
vals =[46.0]        # values to test 

speeds = [0.14,0.42,0.7,0.98,1.96]

start = time.time()

for val in vals:
    val = np.round(val,2)
    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')
    # loop over speeds : 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'/Users/simone/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        print(f'speed = {si}')
        # params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
        # ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  
        os.system(f'python model/plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {val} {stim_name}')

    os.system(f'python model/plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val}')
    
stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))