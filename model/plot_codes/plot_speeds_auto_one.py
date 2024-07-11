from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import os


# define what to compare
save = True
unit = 'space'
#speeds = np.flip([5.0,4.5,4.0,3.5,3.0,2.9,2.8,2.7,2.6])
speeds = [0.14,0.42,0.7,0.98,1.96]
speeds = [0.1,0.2,0.3,0.4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]

#speeds = [0.14,0.7,1.96]
#speeds = [1.0]

filepath = sys.argv[1]
stim_type = sys.argv[2]
param = sys.argv[3]
val = sys.argv[4]
par = f'{param}_{val}'


# load params
fig = plt.figure(figsize=(20,16))

fig.subplots_adjust(hspace=1.2)
gs = fig.add_gridspec(nrows=len(speeds),ncols=2)
label = fig.add_subplot(gs[:,0], frameon = False)
label.set_ylabel('R(t)')
label.tick_params(axis = 'y', colors = 'white')
label.set_xticks([])

ax2 = fig.add_subplot(gs[:,1])

antis = []
cmap = plt.get_cmap('viridis',len(speeds))
for i,s in enumerate(speeds):

    fp = f'{filepath}/{param}/{par}/{stim_type}_{s}'
    print(fp)
    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    plotter = plotting(params,out)
    CELL_GC = int(params['nb_cells']/2)

    if i == 0 :
        ax = fig.add_subplot(gs[i,0])
    else:
        ax = fig.add_subplot(gs[i,0], sharex = ax)


    ax.set_title(f'v {s} mm/s', loc = 'left')
    plotter.plot_one_GC(CELL = CELL_GC,save_one = True,ax = ax,label = f'speed {s}',y = unit, color=cmap(i), linewidth = 5)

    antis.append(params[f'ant_{unit}'][CELL_GC]*1000)

    ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap(i), s = 100)


if unit == 'space':
    ax.set_xlabel('space [mm]')
    ax2.set_ylabel('anticipation [mum]')
if unit == 'time':
    ax.set_xlabel('time [s]')
    ax2.set_ylabel('anticipation [ms]')

ax2.set_xscale('log')
ax2.axhline(0, linestyle = ':')
ax2.plot(speeds,antis, color = 'k')

ax2.set_xlabel('velocity [mum/s]')

fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC} ')


if not os.path.isdir(f'{filepath}/{param}/plots'):
    os.makedirs(f'{filepath}/{param}/plots/')

fig.savefig(f'{filepath}/{param}/plots/RG_speeds_{unit}_{par}.png')

fig.savefig(f'{filepath}/{param}/{par}/RG_speeds_{unit}_{par}.png')
plt.close()