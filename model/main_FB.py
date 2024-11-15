import os
import numpy as np
import time
import matplotlib.pyplot as plt
from run_Reciporcal import run_Reciporcal
from params_FB import  make_params
import pickle
import sys as syt



'''
script to initialize a network with a given parameterset 

simulates the response to an impulse and step stimuli of different lenghts 
plots and saves the outputs 
saves the parameter as a dict
'''

net_name = f'fb_linear_512'


# Simulate response to impule to show model STA
stim_type = 'impulse'


# create output directory for the network 
filepath = f'./output/{net_name}'


# create and save parameter dict
params = make_params(filepath = filepath)


# run simulation
_ = run_Reciporcal(params = params, filepath =f'{filepath}', save_one = True, stim_type=stim_type)  

# load the saved simulation
with open(f'{filepath}/out_{stim_type}', 'rb') as handle:
    out = pickle.load(handle)


# plot impulse response
time = np.arange(0,len(out['VG']))*params['dt']
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(time,out['VB'][0], label = 'VB')
ax.plot(time,out['VA'][0], label = 'VA')

ax = fig.add_subplot(212)
ax.plot(time,out['VG'], label = 'VG')
fig.legend()
fig.savefig(f'{filepath}/plots/STA_fit.svg', format = 'svg')


# Simulate response to steps of different lengths
step_stops = [1.5,2.,3.]
stim_type = 'step'


for step_stop in step_stops:


    _ = run_Reciporcal(params = params, filepath =f'{filepath}', save_one = True, stim_type=stim_type,step_stop = step_stop)  

    # save output
    with open(f'{filepath}/out_{stim_type}', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{filepath}/out_{stim_type}_{step_stop}', 'wb') as handle:
        pickle.dump(out, handle)


    # plot step response 
    
    time = np.arange(0,len(out['VG']))*params['dt']
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(time,out['VB'][0], label = 'VB')
    ax.plot(time,out['VA'][0], label = 'VA')

    ax = fig.add_subplot(212)
    ax.plot(time,out['VG'], label = 'VG')
    fig.legend()
    fig.savefig(f'{filepath}/plots/step_{step_stop}.svg', format = 'svg')



# plot  convolution of step stimulus
fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(time,out['inp'], label = 'convolution')
# ax.plot(time,out['VA'][0], label = 'VA')

ax = fig.add_subplot(212)
ax.plot(time,out['F'], label = 'F')
fig.legend()
fig.savefig(f'{filepath}/plots/stim_step.svg', format = 'svg')