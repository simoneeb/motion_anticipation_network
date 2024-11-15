import os
import numpy as np
import time
from run_Reciporcal import run_Reciporcal
from params_FF import load_params
from loop_speeds_FB import loop_speeds




'''
script to loop over values for one parameter and simulate  model respons eto different speeds
TODO : paralellize
'''


net_name = f'fb_linear_512'
stim_type = 'smooth'


# load parameter of the network 
filepath = f'./output/{net_name}'
params = load_params(filepath,'params')
 
 
# parameter to loop over 
param = 'wBA'                           
vals =np.arange(0.,50., 10) #values for wBA in feedback
vals = [0.,10.,]

# param = 'tau'
# vals = np.linspace(1.,10,10)

# loop over valies
for val in vals:

    val = np.round(val,4)

    params_name = f'{param}/{param}_{val}'
    print(f'{param} = {val}')

    params['betaA'] = 0.0  # make sure no plasticity
    params['wGA'] = 0.0    # make sure no feedforward

    filepathv = f'{filepath}/{param}/{val}'
    if not os.path.isdir(filepathv):
        os.makedirs(filepathv)

    if param == 'tau':
        tauB = params['tauB']
        tauA =  np.round(1/(-val+1/tauB),2)
        params[f'tauA'] = tauA
    else:   
        params[f'{param}'] = val
    loop_speeds(filepathv,params,param) 

    # _ = run_Reciporcal(params = params, filepath = filepathv, save_one = True, stim_type='impulse')  
    # _ = run_Reciporcal(params = params, filepath = filepathv, save_one = True, stim_type='step')  