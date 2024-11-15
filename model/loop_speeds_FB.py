import os
import numpy as np
import time
import sys
from run_Reciporcal import run_Reciporcal
from params_FB import modify_params
from plot_codes.plot_speeds import plot_speeds


def loop_speeds(filepath,params,param):

    '''
    script to loop over different speeds
    '''

    stim_type = 'smooth'

    # speed range
    # speeds = np.round(np.arange(0.12,0.2,0.01),2)
    speeds = np.round(np.arange(0.1,2.0,0.1),2)
    # speeds = np.round(np.arange(2.0,4.0,0.1),2)
    # speeds = np.round(np.arange(4.1,6.0,0.1),2)
    speeds = speeds[::2]
    speeds = np.array([0.8,1.2])


    # loop over speeds 
    for si in speeds:
        print(si)
        stim_name = f'{stim_type}_{si}'
        params = modify_params(params,['speed'],[si])

        _= run_Reciporcal(params = params, filepath =f'{filepath}/{stim_name}', save_one = False,stim_type=stim_type)  

    # plot_speeds(filepath,param,speeds)
