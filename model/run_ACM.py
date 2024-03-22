from stimuli import stim_moving_object_for_2D_net
from connectivity import connectivity
from ACM import ACM
from plotting import plotting
from nonlinearities import N
from utils  import GainF_B,GainF_G, DOG
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import json
import sys



# SPEED = 3.0
# W = 0
# save = True
# filepath = sys.argv[1]
# print(filepath)
# net_name = f'bipolar_pooling_lateral_randpos'
# stim_type = 'smooth'
# param = 'wAB'
# params_name = f'{param}/{param}_{60}'
# stim_name = f'{stim_type}_{4.0}'
# filepath = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}'
# filepath = f'{filepath}/{params_name}/{stim_name}'

def run_ACM(params, filepath = None, save_one = False, measure_n = False, stim_type = 'smooth'):
    if filepath is not None:

        with open(f'{filepath}/params', 'rb') as handle:
            params = pickle.load(handle)



    print('simulation runs')


    # create stimulus
    stimulus_maker = stim_moving_object_for_2D_net(params,
                                                    filepath = filepath)


    if stim_type == 'smooth':
        bar = stimulus_maker.bar_smooth()

    if stim_type == 'onset':
        print('onset')
        bar = stimulus_maker.bar_onset()

    if stim_type == 'reversing':
        bar = stimulus_maker.bar_reversing()

    if stim_type == 'interrupted':
        bar = stimulus_maker.bar_interrupted()
        
    _ = stimulus_maker.load_filter()
    tkern = stimulus_maker.filter_biphasic_norm()
    _,inp = stimulus_maker.OPL()
    stimulus_maker.plot_stim()
    stimulus_maker.plot_kernels()
    params = stimulus_maker.add_params()


    model = ACM(params,inp,filepath=filepath)
    model.make_activity_kernelB()
    model.make_activity_kernelG()
    model.make_GCL_weight_matrix_pooling()
    model.BCL()
    model.GCL()
    out = model.collect_output()
    model.calculate_anticipation()

    params = model.add_params()

    out['inp'] = inp 
    out['stim'] = bar
    # save whole simulation 
    if filepath is not None:
        print('saving')
        print(params.keys())
        with open(f'{filepath}/out', 'wb') as handle:
            pickle.dump(out, handle)
            
        with open(f'{filepath}/params', 'wb') as handle:
            pickle.dump(params, handle)


    # only save maximum
    # with open(f'{filepath}/params.json', 'wb') as handle:
    #     json.dump(params, handle)
