from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import GainF




# define what to compare
net_name = 'selma/selma_net_bipolar_pooling_lateral_w60'
net_name_2 = 'selma/selma_net_bipolar_pooling_lateral_w60_gaincontrol_explore'
net_name_2 = 'selma/selma_net_bipolar_pooling'
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
gs = fig.add_gridspec(nrows=3,ncols=2)
label = fig.add_subplot(gs[:,0], frameon = False)
label.set_ylabel('R(t)',fontsize = fontsize_labels)
label.tick_params(axis = 'y', colors = 'white')
label.set_xticks([])

ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[1,0], sharex = ax0)
ax11 = fig.add_subplot(gs[2,0], sharex = ax0)
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,1])
ax0.set_xlim(-0.1,0.1)

maxis_act = []
antis_lateral = []
n_bipolar = []
n_amacrine = []
cmap_gaincontrol = plt.get_cmap('Reds',len(speeds))
cmap_lateral = plt.get_cmap('Greens',len(speeds))
cmap_response = plt.get_cmap('Blues',len(speeds))
for i,s in enumerate(speeds):

   

   


    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name_2}/{params_name}/{stim_name}{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    time = params['time']
    plotter = plotting(params,out)


    ax0.set_title('Bipolar voltage', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter.plot_one_BC(CELL = CELL_GC,ax = ax0, color=cmap_gaincontrol(i), layer = 0, label = f'v = {s} mm/s', response = 'VB')

    antis_lateral.append(anti)

    ax2.scatter(s, anti, color = cmap_gaincontrol(i), s = 100)

    ax11.set_title('Bipolar Response', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter.plot_one_BC(CELL = CELL_GC,ax = ax11, color=cmap_response(i), layer = 0, label = f'v = {s} mm/s', response = 'RB')

    antis_lateral.append(anti)

    ax2.scatter(s, anti, color = cmap_response(i), s = 100)

    ax1.set_title('Activity', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter.plot_one_BC(CELL = CELL_GC,ax = ax1, color=cmap_lateral(i),  layer = 0, label = f'v = {s} mm/s', response = 'AB')

    maxis_act.append(maxi)
    #ax3.scatter(s, maxi, color = cmap_lateral(i), s = 100)
    





if unit == 'space':
    ax1.set_xlabel('space [mm]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [mum]',fontsize = fontsize_labels)
if unit == 'time':
    ax1.set_xlabel('time [s]',fontsize = fontsize_labels)
    ax2.set_ylabel('anticipation [ms]',fontsize = fontsize_labels)


ax2.axhline(0, linestyle = ':')
# ax2.plot(speeds,antis_gaincontrol, color = 'k')
# ax2.plot(speeds,antis_lateral, color = 'k')

#ax3.axhline(params['wAB'], color = 'k', linestyle = ':')
#ax3.axhline(params['wBA'], color = 'r', linestyle = ':')
#ax3.plot(speeds,maxis_act, color = 'k', label = '')
ax3.scatter(maxis_act,[GainF(m) for m in maxis_act])
range =np.arange(-0.2,3,0.1)
ax3.plot(range,[GainF(m) for m in range])
ax3.legend()

ax3.set_ylabel('Gain(t) ',fontsize = fontsize_labels)
ax3.set_xlabel('max A(t)',fontsize = fontsize_labels)
ax2.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax0.legend(fontsize = fontsize_legend)
ax1.legend(fontsize = fontsize_legend)

fig.suptitle(f'Firing Rate and Anticipation of BC {CELL_GC} via Lateral inhibition with  vs. without Plasticity' ,fontsize = fontsize_labels)

outname = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/RB_speeds_bipolar_acticity_pooling.png'

fig.savefig(outname)


