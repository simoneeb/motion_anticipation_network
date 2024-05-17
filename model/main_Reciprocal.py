import os
import numpy as np
import time
import matplotlib.pyplot as plt
from run_Reciporcal import run_Reciporcal
from params_Reciporcal import load_params, modify_params
import pickle



'''
script to loop over values for one parameter and simulate  model respons eto different speeds
'''


net_name = f'reciprocal_ff_fitted_STA_cell_125'
stim_type = 'impulse'


start = time.time()

stim_name = stim_type
params_name = 'params'
filepath = f'/Users/simoneebert/Documents/Simulations/motion_anticipation_network/{net_name}'

params = load_params(filepath,params_name)
params = modify_params(params, param_names= ['dt'], values=[0.001])
# load params
# if not os.path.isdir(f'{filepath}/impulse'):
#     os.makedirs(f'{filepath}/impulse')
ant_space = run_Reciporcal(params = params, filepath =f'{filepath}', save_one = True,stim_type=stim_type)  


with open(f'{filepath}/out', 'rb') as handle:
    out = pickle.load(handle)

time = np.arange(0,len(out['VG']))*params['dt']

fig = plt.figure()
ax = fig.add_subplot(211)
ax.plot(time,out['VB'][0], label = 'VB')
ax.plot(time,out['VA'][0], label = 'VA')

ax = fig.add_subplot(212)
ax.plot(time,out['VG'], label = 'VG')

fig.savefig(f'{filepath}/STA_fit.png')

x = 0 


#plot response


# # use for moving bar simulation
stim_type = 'smooth'
speeds = [0.14,0.42,0.7,0.98,1.96]

times = []
resps = []
ants = []

for si in speeds:
    stim_name = f'{stim_type}_{si}'
    filepath = f'/Users/simoneebert/Documents/Simulations/motion_anticipation_network/{net_name}/bar'
    params = modify_params(params, param_names = ['speed'], values=[si])
    ant_space = run_Reciporcal(params = params, filepath =f'{filepath}/{params_name}/{stim_name}', save_one = True,stim_type=stim_type)  

os.system(f'python plot_codes/plot_speeds_auto_one.py {filepath} {stim_type} {'params'} {None}')
    
# stop = time.time()

# print('Elapsed time for the entire processing: {:.2f} s'
#       .format(stop - start))