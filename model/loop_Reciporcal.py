import os
import numpy as np
import time

from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params, load_params,modify_params


'''
script to loop over values for one parameter and simulate  model respons eto different speeds
TODO : paralellize
'''


net_name = f'ff_thesis_linear'
stim_type = 'smooth'

param = 'wAB'       # parameter to loop over
vals =[0.0,10.,20.,30.,46.,60.] #[46.0]        # values to test 
#vals =[-0.0005,-0.0007] #[46.0]        # values to test 

speeds = [0.14,0.42,0.7,0.98,1.96]
speeds = [0.1,0.2,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]

start = time.time()

for val in vals:
    val = np.round(val,4)
    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')
    filepath = f'~/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'

home = os.path.expanduser("~")
filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

    # loop over speeds 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        #filepath = f'/Users/simone/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        filepath_p = f'~/Documents/Simulations/motion_anticipation_network/{net_name}'
        print(f'speed = {si}')
        #params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
     
        params = load_params(filepath_p,'params')
        params = modify_params(params, param_names= ['speed',param], values=[si,val])
        ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  
        #os.system(f'python plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {val} {stim_name}')
        

    os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val}')





stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))