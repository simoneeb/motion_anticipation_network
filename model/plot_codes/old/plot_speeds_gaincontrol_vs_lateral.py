from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np




# define what to compare
net_name = 'selma/selma_net_bipolar_pooling_gaincontrol_tauActB0.2'
net_name_2 = 'selma/selma_net_bipolar_pooling_lateral_w60'
params_name = 'initial'
stim_name = 'smooth_'
speeds = [2.22,0.98,0.5,0.25]
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5])
speeds = np.flip([4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1])
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
ax2 = fig.add_subplot(gs[:,1])
ax0.set_xlim(-0.3,0.3)

antis_gaincontrol = []
antis_lateral = []
cmap_gaincontrol = plt.get_cmap('Blues',len(speeds))
cmap_lateral = plt.get_cmap('Reds',len(speeds))

for i,s in enumerate(speeds):

    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    plotter = plotting(params,out)

    ax0.set_title('Gain Control', loc = 'left', fontsize = fontsize_labels)
    plotter.plot_one_GC(CELL = CELL_GC,ax = ax0,y = unit, color=cmap_gaincontrol(i), linewidth = 3, middlecrossing_at_0=True, label = f'v = {s} mm/s')

    antis_gaincontrol.append(params[f'ant_{unit}'][CELL_GC]*1000)

    ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap_gaincontrol(i), s = 100)


    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name_2}/{params_name}/{stim_name}{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    plotter = plotting(params,out)

    ax1.set_title('Lateral Inhibition', loc = 'left', fontsize = fontsize_labels)
    plotter.plot_one_GC(CELL = CELL_GC,ax = ax1,y = unit, color=cmap_lateral(i), linewidth = 3, middlecrossing_at_0=True, label = f'v = {s} mm/s')

    antis_lateral.append(params[f'ant_{unit}'][CELL_GC]*1000)

    ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap_lateral(i), s = 100)


if unit == 'space':
    ax0.set_xlabel('space [mm]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [mum]',fontsize = fontsize_labels)
if unit == 'time':
    ax0.set_xlabel('time [s]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [ms]',fontsize = fontsize_labels)


ax2.axhline(0, linestyle = ':')
ax2.plot(speeds,antis_gaincontrol, color = 'k')
ax2.plot(speeds,antis_lateral, color = 'k')

ax2.set_xscale('log')
ax2.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax0.legend(fontsize = fontsize_legend)
ax1.legend(fontsize = fontsize_legend)

fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC}  with Gain Control vs. Lateral Inhibition' ,fontsize = fontsize_labels)

outname = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/RG_speeds_gaincontrol_lateral.png'

fig.savefig(outname)


