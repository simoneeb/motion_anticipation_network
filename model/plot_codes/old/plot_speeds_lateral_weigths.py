from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np




# define what to compare
params_name = 'initial'
stim_name = 'smooth_'
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5,1.0,0.5])
speeds = np.flip([5.0,4.0,3.0,2.7,2.5,2.4,2.3,2.2,2.1,2.0,1.5])
weights = [40,50,60,70,80,90,100]
save = True


fontsize_labels = 23
fontsize_legend = 15


# load params
unit = 'space'

CELL_GC = 50

fig = plt.figure(figsize=(20,16))

fig.subplots_adjust(hspace=0.9)
gs = fig.add_gridspec(nrows=len(weights),ncols=2)
cmap = plt.get_cmap('plasma',len(weights))


ax1 = fig.add_subplot(gs[:3,1])
ax2 = fig.add_subplot(gs[3:,1], sharex = ax1)
maxis_vals = []
maxis_idxs = []
maxis_speeds = []
maxis_weights = []
for iw,w in enumerate(weights): 
    net_name = f'selma/selma_net_bipolar_pooling_lateral_w{w}'
    antis = []
    for i,s in enumerate(speeds):

        fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}{s}'

        with open(f'{fp}/out', 'rb') as handle:
            out = pickle.load(handle)

        with open(f'{fp}/params', 'rb') as handle:
            params = pickle.load(handle)

        antis.append(params[f'ant_{unit}'][CELL_GC]*1000)
        ax2.scatter(s, params[f'ant_{unit}'][CELL_GC]*1000, color = cmap(iw), s = 100, alpha = 0.5)
    
    ax2.plot(speeds,antis, color = cmap(iw), label = f'w = {w}', alpha = 0.5)
    antis = np.array(antis)

    maxant = antis.argmax()
    maxantval = antis.max()
    sx = speeds[maxant]

    

    maxis_idxs.append(maxant)
    maxis_vals.append(maxantval)
    maxis_speeds.append(sx)
    maxis_weights.append(w)

    ax2.scatter(sx, maxantval, color = cmap(iw), s = 200 )
    ax1.scatter(sx, w, color = cmap(iw), s = 200 )

    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}{sx}'

    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)

    ax = fig.add_subplot(gs[iw,0])
    plotter = plotting(params,out)
    ax.set_title(f'w = {w} Hz, v = {sx} mm/s', loc = 'left' ,fontsize = fontsize_labels)
    plotter.plot_one_GC(CELL = CELL_GC,ax = ax,y = unit, color=cmap(iw), linewidth = 5)

    




if unit == 'space':
    ax2.set_ylabel('anticipation [mum]',fontsize = fontsize_labels)
    ax.set_xlabel('space [mm]',fontsize = fontsize_labels)
if unit == 'time':
    ax2.set_ylabel('anticipation [ms]',fontsize = fontsize_labels)
    ax.set_xlabel('time [s]',fontsize = fontsize_labels)


ax1.plot(maxis_speeds, maxis_weights, color = 'k', linestyle = '--', linewidth = 3)
ax2.axhline(0, linestyle = ':', color = 'k')
ax2.plot(maxis_speeds,maxis_vals, color = 'k', linestyle = '--', linewidth = 3)

ax2.set_xscale('log')
ax1.set_xscale('log')

ax2.set_xlabel('velocity [mum/s]',fontsize = fontsize_labels)
ax1.set_ylabel('weight [Hz]',fontsize = fontsize_labels)
#ax1.set_title('Weight with macimum anticipation time for different speeds',fontsize = fontsize_labels)
#ax2.set_title('Anticipation time for different speeds and weights',fontsize = fontsize_labels)



label = fig.add_subplot(gs[:,0], frameon = False)
label.set_ylabel('R(t)',fontsize = fontsize_labels)
label.tick_params(axis = 'y', colors = 'white')
label.set_xticks([])
#label.set_title('Firing Rates with maximum anticipation time ')
fig.suptitle(f'Firing Rate and Anticipation of GC {CELL_GC} in laterally connected network  with different weights')
fig.legend(fontsize = fontsize_legend)
outname = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/RG_speeds_weights_{unit}.png'

fig.savefig(outname)


