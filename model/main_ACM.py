
import os
import numpy as np
from run_ACM import run_ACM
from params_ACM import make_params


net_name = f'ACM_slow_t2'

stim_type = 'onset'

#loop over parameter
param = 'bar_width'
vals = -1*np.arange(0,100,10)
vals = [0.065,0.065*2]
vals = [0.08]

#speeds = np.flip([4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5,0.4,0.2,0.1])
speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,0.27,0.81,1.62,3.24]
#speeds = [0.27,0.81,1.62,3.24]
speeds = [0.81]

for val in vals:
    val = np.round(val,3)
    params_name = f'{param}/{param}_{val}'

    # loop over speeds : 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/ACM/{net_name}'

        # os.system(f'python params_ACM.py {filepath}/{params_name}/{stim_name} speed {si} {param} {val}')
        # # # # # # #os.system(f'python params.py {filepath}/{params_name}/{stim_name} speed {si} wBA {-1*val} wAB {val}')
        # os.system(f'python run_ACM.py {filepath}/{params_name}/{stim_name} None')
        # # os.system(f'python plot_codes/plot_BC_GC_compare.py {filepath}/{param} {param} {val} {stim_name}')
        params = make_params(param_names = ['speed',param], param_vals=[si,val], filepath= f'{filepath}/{params_name}/{stim_name}')
        ant_space = run_ACM(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  
        os.system(f'python plot_codes/plot_ACM.py {filepath}/{param} {param} {val} {stim_name}')
    
    #os.system(f'python plot_speeds_auto_mean.py {filepath} {stim_type} {param} {val} ')
    os.system(f'python plot_codes/plot_speeds_auto.py {filepath} {stim_type} {param} {val} ')
    # #os.system(f'python plot_pva.py {filepath} {stim_type} {param} {val} ')
    # os.system(f'python plot_speeds_gaincontrol_mechanism.py {filepath} {stim_type} {param} {val}')
    # #os.system(f'python plot_speeds_lateral_mechanism.py {filepath} {stim_type} {param} {val}')

