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

print(params.keys())
# plot responses
#plotter2 = plotting(params,out,filepath = fp)



# fp2 = f'{fp}/{param}/{param}_0/{stim_name}'
# label1 = f'{param} = 0'


# # load params
# with open(f'{fp2}/out', 'rb') as handle:
#     out = pickle.load(handle)

# with open(f'{fp2}/params', 'rb') as handle:
#     params = pickle.load(handle)


# # plot responses
# plotter1 = plotting(params,out,filepath = fp)




CELL_GC = int(params['nb_cells']/2)

# initialize figures 
figsize = (12,9)


# get BC cells that the GC pools from 
rf = DOG(params['pos_rf_mid'],params['pos_rf_mid'][CELL_GC],params['std_GC'], params['std_GC_s'], params['w_GC'])
#rf = DOG(self.pos_rf_mid,self.pos_rf_GC_mid[i],self.std_GC,self.std_GC_surround,self.w)

BC_cells = []
BC_cells_weight = []

for p,val in enumerate(rf):
    if val >= 0.001:
        BC_cells.append(p)
        BC_cells_weight.append(val)

#ax[0.scatter(BC_cells,rf[BC_cells])
BC_cells_weight = np.asarray(BC_cells_weight)

BC_cells_short = BC_cells[0::12]
BC_cells_weight_short = BC_cells_weight[0::12]

fig,ax = plt.subplots(7, sharex = True, figsize = figsize)



ax[-1].set_xlabel('time [ms]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('N(t)')
ax[2].set_ylabel('A(t)')
ax[3].set_ylabel('G(t)')
ax[4].set_ylabel('R(t)')
ax[5].set_ylabel('wR(t)')
ax[6].set_ylabel('V(t)')


ax[0].set_title('BC voltage', loc = 'left')
ax[1].set_title('BC rectified', loc = 'left')
ax[3].set_title('BC gain', loc = 'left')
ax[2].set_title('BC activity', loc = 'left')
ax[4].set_title('BC response', loc = 'left')
ax[5].set_title('BC response weighted', loc = 'left')
ax[6].set_title('GC voltage', loc = 'left')

#ax[1].set_xlim(0.2,0.8)

cmap_pooling = plt.get_cmap('Greys', len(BC_cells_short))
cmap_gaincontrol = plt.get_cmap('Blues', len(BC_cells_short))

# sort weights
cNorm = colors.Normalize(vmin = BC_cells_weight.min()-1, vmax = BC_cells_weight.max())
scalarMap_pooling = cmx.ScalarMappable(norm=cNorm, cmap=cmap_pooling)
scalarMap_gaincontrol = cmx.ScalarMappable(norm=cNorm, cmap=cmap_gaincontrol)

for i,c in enumerate(BC_cells):

    ax[0].plot(out['VB'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[1].plot(out['NB'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[3].plot(out['GB'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[2].plot(out['AB'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[4].plot(out['RB'][c,:], alpha = 0.5,linewidth = 1, color = 'k')


    r =  out['RB'][c] *BC_cells_weight[i] 

    ax[5].plot(r, alpha = 0.5,linewidth = 1, color = 'k')

for i,c in enumerate(BC_cells_short):
    ax[0].plot(out['VB'][c,:], alpha = 1,linewidth = 2, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[1].plot(out['NB'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[2].plot(out['AB'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[3].plot(out['GB'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[4].plot(out['RB'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

    r =  out['RB'][c] *BC_cells_weight_short[i] 

    ax[5].plot(r, alpha = 1,linewidth = 2,  color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

ax[6].plot(out['VG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)

ax[6].legend()
#ax[0.show()
if save == True :
    fig.savefig(f'{fp1}/plots/BC_processing_{stim_name}.png')



fig,ax = plt.subplots(5,1, sharex = True, figsize = figsize)



ax[-1].set_xlabel('time [ms]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('N(t)')
ax[2].set_ylabel('A(t)')
ax[3].set_ylabel('G(t)')
ax[4].set_ylabel('R(t)')


ax[0].set_title('GC voltage', loc = 'left')
ax[1].set_title('GC rectified', loc = 'left')
ax[2].set_title('GC activity', loc = 'left')
ax[2].set_title('GC gain', loc = 'left')
ax[3].set_title('GC response', loc = 'left')

#ax[1].set_xlim(0.2,0.8)


ax[0].plot(out['VG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)
ax[1].plot(out['NG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)
ax[2].plot(out['AG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)
ax[3].plot(out['GG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)
ax[4].plot(out['RG'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)

ax[4].legend()
#ax[0.show()

if save == True :
    fig.savefig(f'{fp1}/plots/GC_processing_{stim_name}.png')




fig,ax = plt.subplots(7, sharex = True, figsize = figsize)



ax[-1].set_xlabel('time [ms]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('N(t)')
ax[2].set_ylabel('A(t)')
ax[3].set_ylabel('G(t)')
ax[4].set_ylabel('R(t)')
ax[5].set_ylabel('wR(t)')
ax[6].set_ylabel('V(t)')


ax[0].set_title('AC voltage', loc = 'left')
ax[1].set_title('AC rectified', loc = 'left')
ax[3].set_title('AC gain', loc = 'left')
ax[2].set_title('AC activity', loc = 'left')
ax[4].set_title('AC response', loc = 'left')
ax[5].set_title('AC response weighted', loc = 'left')
ax[6].set_title('AC voltage', loc = 'left')



for i,c in enumerate(BC_cells):

    ax[0].plot(out['VA'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[1].plot(out['NA'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[3].plot(out['GA'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[2].plot(out['AA'][c,:], alpha = 0.5,linewidth = 1, color = 'k')
    ax[4].plot(out['RA'][c,:], alpha = 0.5,linewidth = 1, color = 'k')


    # r =  out['RA'][c] *BC_cells_weight[i] 

    # ax[5].plot(r, alpha = 0.5,linewidth = 1, color = 'k')

for i,c in enumerate(BC_cells_short):
    ax[0].plot(out['VA'][c,:], alpha = 1,linewidth = 2, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[1].plot(out['NA'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[2].plot(out['AA'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[3].plot(out['GA'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))
    ax[4].plot(out['RA'][c,:], alpha = 1, color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

#     r =  out['RA'][c] *BC_cells_weight_short[i] 

#     ax[5].plot(r, alpha = 1,linewidth = 2,  color = scalarMap_gaincontrol.to_rgba(BC_cells_weight_short[i]))

# ax[6].plot(out['VA'][CELL_GC,:],alpha = 1, color = 'blue', linewidth = 2)

# ax[6].legend()
#ax[0.show()
if save == True :
    fig.savefig(f'{fp1}/plots/AC_processing_{stim_name}.png')