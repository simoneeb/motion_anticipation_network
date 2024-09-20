from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import os




def plot_speeds(filepath,param,speeds):

    unit = 'space'
    stim_type = 'smooth'

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
        stim_name = f'{stim_type}_{s}'

        with open(f'{filepath}/{stim_name}/out_{stim_type}', 'rb') as handle:
            out = pickle.load(handle)

        with open(f'{filepath}/{stim_name}/params', 'rb') as handle:
            params = pickle.load(handle)

        plotter = plotting(params,out)
        CELL_GC = int(params['nb_cells']/2)

        if i == 0 :
            ax = fig.add_subplot(gs[i,0])
        else:
            ax = fig.add_subplot(gs[i,0], sharex = ax)


        ax.set_title(f'v {s} mm/s', loc = 'left')
        plotter.plot_one_GC(CELL = CELL_GC,ax = ax,label = f'speed {s}',y = unit, color=cmap(i), linewidth = 5)

        antis.append(params[f'ant_{unit}'][CELL_GC]*1000)

        ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap(i), s = 100)


    if unit == 'space':
        ax.set_xlabel('space [mm]')
        ax2.set_ylabel('anticipation [mum]')
    if unit == 'time':
        ax.set_xlabel('time [s]')
        ax2.set_ylabel('anticipation [ms]')

    # ax2.set_xscale('log')
    ax2.axhline(0, linestyle = ':')
    ax2.plot(speeds,antis, color = 'k')

    ax2.set_xlabel('velocity [mum/s]')

    fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC} ')

    if not os.path.isdir(f'{filepath}/plots'):
        os.makedirs(f'{filepath}/plots/')
    
    val = params[f'{param}']
    fig.savefig(f'{filepath}/plots/RG_speeds_{unit}_{param}_{val}.png')
