from plotting import plotting
import matplotlib.pyplot as plt
import numpy as np

import pickle
from utils import gaussian, GainF


outname =  f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/new/bipolar_pooling_lateral_OFF_ON_FF/strychnine_'
label1 = '$w_{BA} = 0 Hz$( = strychnine)'
label2 = '$w_{BA} = -60 Hz$ ( = control)'


# define what to compare
net_name = 'new/bipolar_pooling_lateral_OFF_ON_FF'
params_name = 'wBA2/wBA2_0'
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
params_name = 'wBA2/wBA2_-60'
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

for p,val in enumerate(rf):
    if val >= 0.001:
        BC_cells.append(p)

#plt.scatter(BC_cells,rf[BC_cells])


fig,ax = plt.subplots(2,1, sharex = True, figsize = figsize)



ax[1].set_xlabel('time [s]')
ax[0].set_ylabel('V(t)')
ax[1].set_ylabel('R(t)')
ax[0].set_title('Bipolar Responses', loc = 'left')
ax[1].set_title('GC Firing Rate', loc = 'left')




for c in BC_cells:


    plotter1.plot_one_BC(0,c,ax[0],'', response='RB', alpha = 0.5)
    plotter2.plot_one_BC(0,c,ax[0],'', response = 'RB', alpha = 1)


plotter1.plot_one_GC(CELL_GC,ax[1], label1, alpha = 0.5)
plotter2.plot_one_GC(CELL_GC,ax[1], label2, alpha = 1)

plt.legend()

if save == True :
    fig.savefig(f'{outname}_GC_and_BC.png')



