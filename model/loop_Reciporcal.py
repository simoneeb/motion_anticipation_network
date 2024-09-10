import os
import numpy as np
import time

from run_Reciporcal import run_Reciporcal
from params_Reciporcal import make_params, load_params,modify_params


'''
script to loop over values for one parameter and simulate  model respons eto different speeds
TODO : paralellize
'''


net_name = f'fb_linear_600'

stim_type = 'smooth'

param = 'wBA'                           # parameter to loop over
# vals=np.append(0.,np.round(np.logspace(.1,1,10)/100,4)) #[46.0]               # values to test 
vals =np.array([1.,3.,5.,15.,20.,30.]) #[46.0]        # values to test 

# vals = np.arange(0.03,0.14,0.01)

# vals =[.0,10.0] #[46.0]               # values to test 

#vals =[-0.0005,-0.0007] #[46.0]        # values to test 

speeds = np.asarray([0.1,0.2,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
speeds = speeds[::2]
start = time.time()



for val in vals:
    val = np.round(val,4)
    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')

    home = os.path.expanduser("~")
    filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
    params = load_params(filepath,'params')
    params['betaA'] = 0.0
    params['wGA'] = 0.0
    # params['wBA'] = 10.0
    params['wAB'] = 10.0
    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    # loop over speeds 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'

        #filepath = f'/Users/simone/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        print(f'speed = {si}')
        params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
        ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  

    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}', save_one = True, stim_type='impulse')  
    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}', save_one = True, stim_type='step')  


    os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {val}')




stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))