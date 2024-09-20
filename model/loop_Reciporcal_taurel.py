import os
import numpy as np
import time

from run_Reciporcal import run_Reciporcal
from model.params_FB import make_params, load_params,modify_params


'''
script to loop over values for one parameter and simulate  model respons eto different speeds
TODO : paralellize
'''


net_name = f'ff_linear'

stim_type = 'smooth'

param = 'tauA'       # parameter to loop over

#vals =[-0.0005,-0.0007] #[46.0]        # values to test 

vals = np.linspace(1.,10,10)


x = 5.8


speeds = np.asarray([0.1,0.2,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
speeds = speeds[::2]
start = time.time()

for val in vals:

    tauB = 0.08
    x = np.round(val,2)
    tauA =  np.round(1/(-x+1/tauB),2)

    params_name = f'{param}/{param}_{tauA}'
    print(f'{param} = {tauA},x =  {x}')

    home = os.path.expanduser("~")
    filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
    params = load_params(filepath,'params')

    if not os.path.isdir(filepath):
        os.makedirs(filepath)

    # loop over speeds 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'

        #filepath = f'/Users/simone/Documents/Simulations/motion_anticipation_network/Loops/{net_name}'
        print(f'speed = {si}')
        params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
     
        params = modify_params(params, param_names= ['speed',param,'tauB'], values=[si,tauA,tauB])
        # ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  


        os.system(f'python plot_codes/plot_Reciporcal_one.py {filepath}/{param} {param} {tauA} {stim_name}')

    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}', save_one = True, stim_type='impulse')  
    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}', save_one = True, stim_type='step')  


    os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {param} {tauA}')





stop = time.time()

print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))