from plotting import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

import pickle
from utils import gaussian, GainF


outname =  f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/polling_vs_lateral_'
label1 = 'pooling'
label2 = '+ lateral inhibition'


# define what to compare
net_name = 'selma/selma_net_bipolar_pooling'
params_name = 'initial'
stim_name = 'smooth_3.0'
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}'
save = True


# load params
with open(f'{fp}/out', 'rb') as handle:
    out = pickle.load(handle)

with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)


# plot responses
plotter1 = plotting(params,out)



# define what to compare
net_name = 'selma/selma_net_bipolar_pooling_lateral_w80'
params_name = 'initial'
stim_name = 'smooth_3.0'
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/{net_name}/{params_name}/{stim_name}'
save = True


# load params
with open(f'{fp}/out', 'rb') as handle:
    out = pickle.load(handle)

with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)


# plot responses
plotter2 = plotting(params,out)




CELL_BC = 15
CELL_GC = 50

# initialize figures 
figsize = (8,6)


# BC CELLS

fig,ax = plt.subplots(2,1, sharex = True, figsize = figsize)


ax[0].set_ylabel('s(t)')
ax[0].set_title('Stimulus')


ax[1].set_xlabel('time [s]')
ax[1].set_ylabel('V(t)')
ax[1].set_title('BC Responses')


plotter1.plot_one_stim(CELL_BC,ax[0])
plotter1.plot_one_BC(0,CELL_BC,ax[1],label1)
plotter2.plot_one_BC(0,CELL_BC,ax[1],label2)


plt.legend()

if save == True :
    fig.savefig(f'/{outname}_BC.png')


# AC CELLS

fig,ax = plt.subplots(2,1, sharex = True, figsize = figsize)

ax[0].set_ylabel('s(t)')
ax[0].set_title('Stimulus')


ax[1].set_xlabel('time [s]')
ax[1].set_ylabel('V(t)')
ax[1].set_title('AC Responses')

plotter1.plot_one_stim(CELL_BC,ax[0])
plotter1.plot_one_AC(1,CELL_BC,ax[1], label1)
plotter2.plot_one_AC(1,CELL_BC,ax[1], label2)


plt.legend()
if save == True :
    fig.savefig(f'{outname}_AC.png')




# GC CELLS

fig,ax = plt.subplots(1,1, sharex = True, figsize = (10,4.5))
fontsize = 17


ax.set_xlabel('distance from rf center [mm]', fontsize = fontsize)
ax.set_xlim(-1,1)
#ax[0].set_ylabel('V(t)')
ax.set_ylabel('R(t)', fontsize = fontsize)
ax.set_title('Simulation', loc ='left',fontsize = fontsize)

plotter1.plot_one_GC(CELL_GC,ax, label1,linestyle = ':', y = 'neural image', linewidth =3, alpha = 1)
plotter2.plot_one_GC(CELL_GC,ax,label2,linestyle = '-', y = 'neural image', linewidth = 3, alpha = 1)
#plotter1.plot_one_GC(CELL_GC,ax[1], label1)
#plotter2.plot_one_GC(CELL_GC,ax[1], label2)


fig.legend(fontsize = 14)

x = 0
if save == True :
    fig.savefig(f'{outname}_GC.png')







#  one GC CELL with all its inputs 

# get BC cells that the GC pools from 
rf = gaussian(params['pos_rf_mid'],params['pos_rf_GC_mid'][CELL_GC],params['std_GC'])
BC_cells = []
BC_cells_weight = []

for p,val in enumerate(rf):
    if val >= 0.001:
        BC_cells.append(p)
        BC_cells_weight.append(val)

#plt.scatter(BC_cells,rf[BC_cells])
BC_cells_weight = np.asarray(BC_cells_weight)

fig,ax = plt.subplots(2,1, sharex = True, figsize = figsize)



ax[1].set_xlabel('time [s]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('R(t)')
ax[0].set_title('Bipolar Responses', loc = 'left')
ax[1].set_title('GC Firing Rate', loc = 'left')

ax[1].set_xlim(0.2,0.8)

cmap_pooling = plt.get_cmap('Greys', len(BC_cells))
cmap_gaincontrol = plt.get_cmap('Reds', len(BC_cells))

# sort weights
cNorm = colors.Normalize(vmin = BC_cells_weight.min()-1, vmax = BC_cells_weight.max())
scalarMap_pooling = cmx.ScalarMappable(norm=cNorm, cmap=cmap_pooling)
scalarMap_gaincontrol = cmx.ScalarMappable(norm=cNorm, cmap=cmap_gaincontrol)

for i,c in enumerate(BC_cells):

    # take the color 

    plotter1.plot_one_BC(0,c,ax[0],'', response='RB', alpha = 0.5, color = scalarMap_pooling.to_rgba(BC_cells_weight[i]))
    plotter2.plot_one_BC(0,c,ax[0],'', response = 'RB', alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight[i]))


plotter1.plot_one_GC(CELL_GC,ax[1], label1, alpha = 1, color = 'grey', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[1], label2, alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight.max()), linewidth = 2)

plt.legend()

if save == True :
    fig.savefig(f'{outname}_GC_and_BC.png')



print(scalarMap_gaincontrol.to_rgba(BC_cells_weight.max()))