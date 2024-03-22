
import os
import numpy as np


net_name = f'Laplacian_slow'

stim_type = 'smooth'

#loop over parameter
param = 'w_GC'
vals = -1*np.arange(0,100,10)
vals = [0]


speeds = np.flip([4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5,0.4,0.2,0.1])
speeds = [0.1,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0,3.0]
speeds = [0.27,0.81,1.62,3.24]
speeds = [0.81]


for val in vals:
    val = np.round(val,2)
    params_name = f'{param}/{param}_{val}'

    # loop over speeds : 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Laplacian/{net_name}'

        os.system(f'python params_Laplacian.py {filepath}/{params_name}/{stim_name} speed {si} {param} {val}')
        # # # # #os.system(f'python params.py {filepath}/{params_name}/{stim_name} speed {si} wBA {-1*val} wAB {val}')
        os.system(f'python run_Laplacian.py {filepath}/{params_name}/{stim_name} None')
        # os.system(f'python plot_codes/plot_BC_GC_compare.py {filepath}/{param} {param} {val} {stim_name}')
        os.system(f'python plot_codes/plot_Laplacian.py {filepath}/{param} {param} {val} {stim_name}')
    
    # os.system(f'python plot_speeds_auto_mean.py {filepath} {stim_type} {param} {val} ')
    # os.system(f'python plot_speeds_auto.py {filepath} {stim_type} {param} {val} ')
    # #os.system(f'python plot_pva.py {filepath} {stim_type} {param} {val} ')
    # os.system(f'python plot_speeds_gaincontrol_mechanism.py {filepath} {stim_type} {param} {val}')
    # #os.system(f'python plot_speeds_lateral_mechanism.py {filepath} {stim_type} {param} {val}')

