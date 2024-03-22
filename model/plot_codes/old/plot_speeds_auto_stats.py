from plotting import plotting
from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import os



# define what to compare
save = True
unit = 'space'
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5,0.4,0.2,0.1])
speeds = np.flip([5.0,4.5,4.0,3.5,3.0,2.9,2.8,2.7,2.6])

#speeds = [4.0]


filepath = sys.argv[1]
stim_type = sys.argv[2]
param = sys.argv[3]
val = sys.argv[4]
par = f'{param}_{val}'


# load params

CELL_GC = 50

fig = plt.figure(figsize=(20,16))

fig.subplots_adjust(hspace=1.2)
gs = fig.add_gridspec(nrows=len(speeds),ncols=2)
label = fig.add_subplot(gs[:,0], frameon = False)
label.set_ylabel('R(t)')
label.tick_params(axis = 'y', colors = 'white')
label.set_xticks([])

ax2 = fig.add_subplot(gs[:,1])


antis = []
stds = []
cmap = plt.get_cmap('viridis',len(speeds))

for i,s in enumerate(speeds):
    

    fp = f'{filepath}/{param}/{par}/{stim_type}_{s}'

    # with open(f'{fp}/out', 'rb') as handle:
    #     out = pickle.load(handle)
    
    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)    
        
    with open(f'{fp}/stats', 'rb') as handle:
        stats = pickle.load(handle)

    anti_mean = stats['RG']['anti_mean'][CELL_GC,-1]*params['speed']
    print(anti_mean)
    anti_std = stats['RG']['anti_std'][CELL_GC,-1]*params['speed']

    print(fp)
    print(params.keys())
    # ants = np.asarray(params[f'ant_{unit}'][10:-10])*1000
    # ants_mean = np.mean(ants)
    # ants_std = np.std(ants)

    plotter = plotting(params,out= None,stats = stats, filepath = filepath)

    ax = fig.add_subplot(gs[i,0])
    ax.set_title(f'v {s} mm/s', loc = 'left')
    plotter.plot_stats(CELL = CELL_GC, ax = ax,label = f'speed {s}',y = unit, color=cmap(i), linewidth = 5)
  
    antis.append(anti_mean)
    stds.append(anti_std)

    ax2.scatter(s, anti_mean, color = cmap(i), s = 100)


if unit == 'space':
    ax.set_xlabel('space [mm]')
    ax2.set_ylabel('anticipation [mum]')
if unit == 'time':
    ax.set_xlabel('time [s]')
    ax2.set_ylabel('anticipation [ms]')

ax2.set_xscale('log')
ax2.axhline(0, linestyle = ':')
ax2.plot(speeds,antis, color = 'k')
ax2.errorbar(speeds,antis,stds, color = 'k')

ax2.set_xlabel('velocity [mum/s]')

fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC} ')


if not os.path.isdir(f'{filepath}/{param}/plots'):
    os.makedirs(f'{filepath}/{param}/plots/')

fig.savefig(f'{filepath}/{param}/plots/RG_speeds_stats_{unit}_{par}.png')


fig.savefig(f'{filepath}/{param}/{par}/RG_speeds_stats_{unit}_{par}.png')
