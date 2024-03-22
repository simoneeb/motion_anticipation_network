from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np




# define what to compare
net_name = 'selma/bipolar_pooling_lateral_plastic/betaA'
net_name_2 = 'selma/bipolar_pooling_lateral_plastic/betaA'
params_name = 'betaA_0'
params_name2 = 'betaA_240'
stim_name = 'smooth_'
speeds = [2.22,0.98,0.5,0.25]
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5])
speeds = np.flip([4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1])
speeds = np.flip([5.0,4.5,4.0,3.5,3.0,2.9,2.8,2.7,2.6])

save = True


fontsize_labels = 23
fontsize_legend = 15

# load params
unit = 'space'

CELL_GC = 50

fig = plt.figure(figsize=(20,16))

fig.subplots_adjust(hspace=0.2)
gs = fig.add_gridspec(nrows=2,ncols=2)
label = fig.add_subplot(gs[:,0], frameon = False)
label.set_ylabel('R(t)',fontsize = fontsize_labels)
label.tick_params(axis = 'y', colors = 'white')
label.set_xticks([])

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0], sharex = ax0)
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])
ax0.set_xlim(-0.3,0.3)

antis_gaincontrol = []
antis_lateral = []
n_bipolar = []
n_amacrine = []
cmap_gaincontrol = plt.get_cmap('Reds',len(speeds))
cmap_lateral = plt.get_cmap('Greens',len(speeds))

for i,s in enumerate(speeds):

    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    plotter1 = plotting(params,out)

    ax0.set_title('Lateral Inhibition', loc = 'left', fontsize = fontsize_labels)
    plotter1.plot_one_GC(CELL = CELL_GC,ax = ax0,y = unit, color=cmap_gaincontrol(i), linewidth = 3, middlecrossing_at_0=True, label = f'v = {s} mm/s')

    antis_gaincontrol.append(params[f'ant_{unit}'][CELL_GC]*1000)

    ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap_gaincontrol(i), s = 100)


    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name_2}/{params_name2}/{stim_name}{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    plotter2 = plotting(params,out)

    ax1.set_title('Lateral Inhibition + Plasticity', loc = 'left', fontsize = fontsize_labels)
    plotter2.plot_one_GC(CELL = CELL_GC,ax = ax1,y = unit, color=cmap_lateral(i), linewidth = 3, middlecrossing_at_0=True, label = f'v = {s} mm/s')

    antis_lateral.append(params[f'ant_{unit}'][CELL_GC]*1000)

    ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap_lateral(i), s = 100)


    #get occupancy pipolar
    n_speed = np.mean(out['res'][0]['n'][CELL_GC])
    ax3.scatter(s, n_speed, color = cmap_lateral(i), s = 100)
    n_bipolar.append(n_speed)
    n_speed = np.mean(out['res'][1]['n'][CELL_GC])
    n_amacrine.append(n_speed)
    ax3.scatter(s, n_speed, color = cmap_lateral(i), s = 100)





if unit == 'space':
    ax0.set_xlabel('space [mm]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [mum]',fontsize = fontsize_labels)
if unit == 'time':
    ax0.set_xlabel('time [s]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [ms]',fontsize = fontsize_labels)


ax2.axhline(0, linestyle = ':')
ax2.set_xscale('log')
ax3.set_xscale('log')

ax2.plot(speeds,antis_gaincontrol, color = 'k')
ax2.plot(speeds,antis_lateral, color = 'k')

#ax3.axhline(params['wAB'], color = 'k', linestyle = ':')
#ax3.axhline(params['wBA'], color = 'r', linestyle = ':')
ax3.plot(speeds,n_bipolar, color = 'k', label = 'bipolar')
ax3.plot(speeds,n_amacrine, color = 'r', label = 'amacrine')
ax3.legend()

ax3.set_ylabel('n ',fontsize = fontsize_labels)
ax3.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax2.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax0.legend(fontsize = fontsize_legend)
ax1.legend(fontsize = fontsize_legend)

fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC} via Lateral inhibition with  vs. without Plasticity' ,fontsize = fontsize_labels)

outname = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma//bipolar_pooling_lateral_plastic/RG_speeds_fixed_plastic.png'

fig.savefig(outname)


