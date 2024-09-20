import os
import numpy as np
import time
import sys
from run_Reciporcal import run_Reciporcal
from plot_codes.plot_speeds import plot_speeds


def loop_speeds(filepath,params,param):

    '''
    script to loop over different speeds
    '''

    stim_type = 'smooth'

    # slow speed range
    # speeds = np.arange(0.01,0.1,0.01)
    speeds = np.arange(0.1,2.0,0.1)
    speeds = speeds[::2]


    # loop over speeds 
    for si in speeds:
        stim_name = f'{stim_type}_{si}'
        params['speed'] = si
        # _= run_Reciporcal(params = params, filepath =f'{filepath}/{stim_name}', save_one = True,stim_type=stim_type)  

    # plot_speeds(filepath,param,speeds)
