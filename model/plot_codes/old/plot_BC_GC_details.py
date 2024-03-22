from plotting import plotting
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np
import sys

import pickle
from utils import gaussian, GainF, DOG



# net_name = f'Bipolar_Pooling_OPL'
# stim_type = 'smooth'
# param = 'w_GC'
# val = 0
# si = 0.27
# params_name = f'{param}/{param}_{val}'
# stim_name = f'{stim_type}_{si}'
# fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/newparams/{net_name}/{param}'

fp = sys.argv[1]
param = sys.argv[2]
val = sys.argv[3]
stim_name = sys.argv[4]

label2 = f'{param} = {val}'

fp1 = f'{fp}/{param}_{val}/{stim_name}'
save = True
print(fp)

# load params
with open(f'{fp1}/out', 'rb') as handle:
    out = pickle.load(handle)

with open(f'{fp1}/params', 'rb') as handle:
    params = pickle.load(handle)


# plot responses
plotter2 = plotting(params,out,filepath = fp)



# fp2 = f'{fp}/{param}/{param}_0/{stim_name}'
# label1 = f'{param} = 0'


# # load params
# with open(f'{fp2}/out', 'rb') as handle:
#     out = pickle.load(handle)

# with open(f'{fp2}/params', 'rb') as handle:
#     params = pickle.load(handle)


# # plot responses
# plotter1 = plotting(params,out,filepath = fp)




CELL_GC = 300

# initialize figures 
figsize = (8,6)


# get BC cells that the GC pools from 
rf = DOG(params['pos_rf_mid'],params['pos_rf_GC_mid'][CELL_GC],params['std_GC'], params['std_GC_s'], params['w_GC'])
#rf = DOG(self.pos_rf_mid,self.pos_rf_GC_mid[i],self.std_GC,self.std_GC_surround,self.w)

BC_cells = []
BC_cells_weight = []

for p,val in enumerate(rf):
    if val >= 0.001:
        BC_cells.append(p)
        BC_cells_weight.append(val)

#plt.scatter(BC_cells,rf[BC_cells])
BC_cells_weight = np.asarray(BC_cells_weight)

BC_cells_short = BC_cells[0::12]
BC_cells_weight_short = BC_cells_weight[0::12]

fig,ax = plt.subplots(6,1, sharex = True, figsize = figsize)



ax[-1].set_xlabel('time [ms]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('N(t)')
ax[2].set_ylabel('G(t)')
ax[3].set_ylabel('R(t)')
ax[4].set_ylabel('wR(t)')
ax[5].set_ylabel('V(t)')


ax[0].set_title('Bipolar voltage', loc = 'left')
ax[1].set_title('Bipolar rectified', loc = 'left')
ax[2].set_title('Bipolar Gain', loc = 'left')
ax[3].set_title('Bipolar response', loc = 'left')
ax[4].set_title('Bipolar response weighted', loc = 'left')
ax[5].set_title('GC voltage', loc = 'left')

#ax[1].set_xlim(0.2,0.8)

cmap_pooling = plt.get_cmap('Greys', len(BC_cells_short))
cmap_gaincontrol = plt.get_cmap('Blues', len(BC_cells_short))

# sort weights
cNorm = colors.Normalize(vmin = BC_cells_weight.min()-1, vmax = BC_cells_weight.max())
scalarMap_pooling = cmx.ScalarMappable(norm=cNorm, cmap=cmap_pooling)
scalarMap_gaincontrol = cmx.ScalarMappable(norm=cNorm, cmap=cmap_gaincontrol)

for i,c in enumerate(BC_cells):

    plotter2.plot_one_BC(0,c,ax[0],'', response='VB', alpha = 0.5,linewidth = 1, color = 'k')
    plotter2.plot_one_BC(0,c,ax[1],'', response='NB', alpha = 0.5,linewidth = 1, color = 'k')
    plotter2.plot_one_BC(0,c,ax[2],'', response='GB', alpha = 0.5,linewidth = 1, color = 'k')
    plotter2.plot_one_BC(0,c,ax[3],'', response='RB', alpha = 0.5,linewidth = 1, color = 'k')

    r =  out['RB'][c] *BC_cells_weight[i] 

    ax[4].plot(r, alpha = 0.5,linewidth = 1, color = 'k')

for i,c in enumerate(BC_cells_short):
    plotter2.plot_one_BC(0,c,ax[0],'', response='VB', alpha = 1,linewidth = 2, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    plotter2.plot_one_BC(0,c,ax[1],'', response='NB', alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    plotter2.plot_one_BC(0,c,ax[2],'', response='GB', alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    plotter2.plot_one_BC(0,c,ax[3],'', response='RB', alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

    r =  out['RB'][c] *BC_cells_weight_short[i] 

    ax[4].plot(r, alpha = 1,linewidth = 2,  color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

plotter2.plot_one_GC(CELL_GC,ax[5], label2,response='VG' ,alpha = 1, color = 'blue', linewidth = 2)

plt.legend()
#plt.show()
if save == True :
    fig.savefig(f'{fp1}/plots/BC_processing_{stim_name}.png')



fig,ax = plt.subplots(5,1, sharex = True, figsize = figsize)



ax[-1].set_xlabel('time [ms]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('N(t)')
ax[2].set_ylabel('R(t)')
ax[3].set_ylabel('G(t)')


ax[0].set_title('GC voltage', loc = 'left')
ax[1].set_title('GC rectified', loc = 'left')
ax[2].set_title('GC response', loc = 'left')
ax[3].set_title('GC gain', loc = 'left')

#ax[1].set_xlim(0.2,0.8)


plotter2.plot_one_GC(CELL_GC,ax[0], label2,response='VG' ,alpha = 1, color = 'blue', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[1], label2,response='NG' ,alpha = 1, color = 'blue', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[2], label2,response='RG' ,alpha = 1, color = 'blue', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[2], label2,response='RG2' ,alpha = 1, color = 'blue', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[3], label2,response='GG' ,alpha = 1, color = 'blue', linewidth = 2)
plotter2.plot_one_GC(CELL_GC,ax[4], label2,response='AG' ,alpha = 1, color = 'blue', linewidth = 2)

plt.legend()
#plt.show()

if save == True :
    fig.savefig(f'{fp1}/plots/GC_processing_{stim_name}.png')