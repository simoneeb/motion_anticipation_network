from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utils import GainF
import sys



# define what to compare
save = True
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5])
speeds = np.flip([5.0,4.5,4.0,3.5,3.0,2.9,2.8,2.7,2.6])

#speeds = [4.0]


filepath = sys.argv[1]
stim_type = sys.argv[2]
param = sys.argv[3]
val = sys.argv[4]
par = f'{param}_{val}'




fontsize_labels = 23
fontsize_legend = 15

# load params
unit = 'space'

CELL_GC = 50

fig = plt.figure(figsize=(20,16))

fig.subplots_adjust(hspace=0.4, wspace = 0.5)
gs = fig.add_gridspec(nrows=3,ncols=6)


ax0 = fig.add_subplot(gs[0,0:3])
ax1 = fig.add_subplot(gs[1,0:3], sharex = ax0)
ax2 = fig.add_subplot(gs[2,0:3], sharex = ax0)
ax0.set_xlim(-0.1,1)


ax10 = fig.add_subplot(gs[0,3:])

ax11 = fig.add_subplot(gs[1,3:])


ax12 = fig.add_subplot(gs[2,3:])



maxis_act = []

antis_pooling = []
maxis_pooling = []

antis_gaincontrol = []
maxis_gaincontrol = []


cmap_pooling = plt.get_cmap('Greys',len(speeds))
cmap_gaincontrol = plt.get_cmap('Reds',len(speeds))
cmap_activity = plt.get_cmap('Greens',len(speeds))



for i,s in enumerate(speeds):
    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/bipolar_pooling/initial/initial_0/{stim_type}_{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    time = params['time']
    plotter = plotting(params,out)

    ax0.set_title('Bipolar Voltage without Amacrine', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter.plot_one_BC(CELL = CELL_GC,ax = ax0, color=cmap_pooling(i), layer = 0, label = f'v = {s} mm/s', response = 'VB', middlecrossing_at_0=True)

    antis_pooling.append(anti)
    maxis_pooling.append(anti)

    ax10.set_title('Maximum Bipolar Voltage/Response', loc = 'left', fontsize = fontsize_labels)
    ax10.scatter(s, maxi, color = cmap_pooling(i), s = 100)

    ax12.set_title('Anticipation Bipolar Voltage/Response', loc = 'left', fontsize = fontsize_labels)
    ax12.scatter(s, anti, color = cmap_pooling(i), s = 100)


    fp = f'{filepath}/{param}/{par}/{stim_type}_{s}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    time = params['time']
    plotter2 = plotting(params,out)


    ax1.set_title('Amacrine voltage ', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter2.plot_one_AC(CELL = CELL_GC,ax = ax1, color=cmap_activity(i), layer = 1, label = f'v = {s} mm/s', middlecrossing_at_0=True)

    maxis_act.append(maxi)
    ax11.scatter(s, maxi, color = cmap_activity(i), s = 100)

    ax2.set_title('Bipolar Response with Lateral Inhibition', loc = 'left', fontsize = fontsize_labels)
    anti,maxi = plotter2.plot_one_BC(CELL = CELL_GC,ax = ax2, color=cmap_gaincontrol(i), layer = 0, label = f'v = {s} mm/s', response = 'RB', middlecrossing_at_0=True)

    antis_gaincontrol.append(anti)
    maxis_gaincontrol.append(maxi)

    ax10.scatter(s, maxi, color = cmap_gaincontrol(i), s = 100)
    ax12.scatter(s, anti, color = cmap_gaincontrol(i), s = 100)
 

    #ax3.scatter(s, maxi, color = cmap_lateral(i), s = 100)



ax11.set_title('Max Amacrine', fontsize = fontsize_labels, loc = 'left')




ax0.set_ylabel('V(t) ',fontsize = fontsize_labels)
ax1.set_ylabel('A(t) ',fontsize = fontsize_labels)
ax2.set_ylabel('R(t) ',fontsize = fontsize_labels)

ax10.set_ylabel('max ',fontsize = fontsize_labels)
ax11.set_ylabel('max',fontsize = fontsize_labels)
ax12.set_ylabel('anticipation [ms]',fontsize = fontsize_labels)


ax2.set_xlabel('time [s]',fontsize = fontsize_labels)
ax10.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax11.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax12.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)


#ax2.axhline(0, linestyle = ':')
# ax2.plot(speeds,antis_gaincontrol, color = 'k')
# ax2.plot(speeds,antis_lateral, color = 'k')

#ax3.axhline(params['wAB'], color = 'k', linestyle = ':')
#ax3.axhline(params['wBA'], color = 'r', linestyle = ':')
#ax3.plot(speeds,maxis_act, color = 'k', label = '')



ax0.legend(fontsize = fontsize_legend)
ax1.legend(fontsize = fontsize_legend)
ax2.legend(fontsize = fontsize_legend)

fig.suptitle(f'Anticipation Mechanism via Gain Control in Bipolars' ,fontsize = fontsize_labels)

fig.savefig(f'{filepath}/{param}/plots/RB_speeds_bipolar_{par}.png')
fig.savefig(f'{filepath}/{param}/{par}/RB_speeds_bipolar_{par}.png')
