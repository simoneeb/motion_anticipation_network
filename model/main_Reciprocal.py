import os
import numpy as np
import time
import matplotlib.pyplot as plt
from run_Reciporcal import run_Reciporcal
from params_Reciporcal import load_params, modify_params, make_params
import pickle
import sys as syt



'''
script to loop over values for one parameter and simulate  model respons eto different speeds
'''

net_name = f'fb_linear_dt'


# Simulate response to impule to show model STA
stim_type = 'impulse'


start = time.time()

stim_name = stim_type
params_name = 'params'

home = os.path.expanduser("~")
filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

params = make_params(filepath = filepath)
#params = load_params(filepath,params_name)
params = modify_params(params, param_names= ['dt'], values=[0.001])
# load params
# if not os.path.isdir(f'{filepath}/impulse'):
#     os.makedirs(f'{filepath}/impulse')
ant_space = run_Reciporcal(params = params, filepath =f'{filepath}', save_one = True, stim_type=stim_type)  
print(params['saving_range'])

with open(f'{filepath}/out_{stim_type}', 'rb') as handle:
    out = pickle.load(handle)

time = np.arange(0,len(out['VG']))*params['dt']

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(time,out['VB'][0], label = 'VB')
ax.plot(time,out['VA'][0], label = 'VA')

ax = fig.add_subplot(212)
ax.plot(time,out['VG'], label = 'VG')
fig.legend()
fig.savefig(f'{filepath}/plots/STA_fit.svg', format = 'svg')

x = 0 


# Simulate response to steps of different lengths
step_stops = [1.5,2.,3.]
stim_type = 'step'


stim_name = stim_type
params_name = 'params'
filepath = f'~/Documents/Simulations/motion_anticipation_network/{net_name}'

home = os.path.expanduser("~")
filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

params = make_params(filepath = filepath)
#params = load_params(filepath,params_name)
params = modify_params(params, param_names= ['dt'], values=[0.001])
# load params
# if not os.path.isdir(f'{filepath}/impulse'):
#     os.makedirs(f'{filepath}/impulse')

for step_stop in step_stops:
    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}', save_one = True, stim_type=stim_type,step_stop = step_stop)  
    print(params['saving_range'])

    with open(f'{filepath}/out_{stim_type}', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{filepath}/out_{stim_type}_{step_stop}', 'wb') as handle:
        pickle.dump(out, handle)

    time = np.arange(0,len(out['VG']))*params['dt']

    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.plot(time,out['VB'][0], label = 'VB')
    ax.plot(time,out['VA'][0], label = 'VA')

    ax = fig.add_subplot(212)
    ax.plot(time,out['VG'], label = 'VG')
    fig.legend()
    fig.savefig(f'{filepath}/plots/step_{step_stop}.svg', format = 'svg')



fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(time,out['inp'], label = 'convolution')
# ax.plot(time,out['VA'][0], label = 'VA')

ax = fig.add_subplot(212)
ax.plot(time,out['F'], label = 'F')
fig.legend()
fig.savefig(f'{filepath}/plots/stim.svg', format = 'svg')
print("ALL SAVED")
# print(params)

# plot response
# stim_type = 'smooth'
# speeds = [0.14,0.42,0.7,0.98,1.96]
# speeds = [0.1,0.2,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]

# times = []
# resps = []
# ants = []

# for si in speeds:
#     stim_name = f'{stim_type}_{si}'
#     filepath = f'~/Documents/Simulations/motion_anticipation_network/{net_name}/bar'
#     params = modify_params(params, param_names = ['speed'], values=[si])
#     ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  

# os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {'params'} {None}')

# TODO fix save path
# stop = time.time()

# print('Elapsed time for the entire processing: {:.2f} s'
#       .format(stop - start))