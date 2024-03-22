from plotting import plotting
from plotting import plotting
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import os
import pandas as pd
import seaborn as sns


#speeds = np.flip([5.0,4.5,4.0,3.5,3.0,2.9,2.8,2.7,2.6])
speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,2.0]

weights = np.arange(0,200,10)


# filepath = sys.argv[1]
stim_type = 'smooth'
# param = sys.argv[3]
# val = sys.argv[4]
# par = f'{param}_{val}'



# load dataframe 
response_data = pd.read_csv('/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/Reciporcal_fitted_linear/noGCGainControl/responses_2.csv')


# load dataframe 
df = pd.read_csv('/user/sebert/home/Documents/Simulations/motion/anticipation_1D/Reciporcal/Reciporcal_fitted_linear/noGCGainControl/anticipation_data_2.csv')


print(df.columns)


# calculate anticipation with respect to V_drive
df['ant_RG_drive_time']  =  df['peak_drive'] - df['peak_RG'] 
df['ant_RG_drive_space'] = df['ant_RG_drive_time']* df['speed']

# calculate anticipation with respect to 
df['ant_RG_bar_time']  = df['tp_rf_GC_mid'] - df['peak_RG'] 
df['ant_RG_bar_space'] = df['ant_RG_bar_time']* df['speed']


w_fix = 'wBA'
w_plot = 'wAB'
# multiple heatmaps, keeping w+ fixed and changing w-
dfgrouped = df.groupby([w_fix])

var = 'ant_RG_bar_space'

fig = plt.figure()
fig.tight_layout()
fig.suptitle(f'{var} for {w_plot} and speeds, keeping {w_fix} fixed')


fig2 = plt.figure()
fig2.tight_layout()
fig2.suptitle(f'{var} against speed  for {w_plot}, keeping {w_fix} fixed')
#weights = [0.0,10.0,50.0,100.0]
weights = dfgrouped[w_plot].unique()[0]

for i,w in enumerate(weights):
    print(w)
    ax = fig.add_subplot(4,4,i+1)
    ax.set_title(f'{w_fix} = {w}')
    g = dfgrouped.get_group(w)
    # plot heatmap for beta
    g_heatmap = g.pivot(w_plot,'speed',var)
    sns.heatmap(g_heatmap, ax = ax, cmap = 'bwr',  vmin=-0.2, vmax=0.2)

    subg = g.groupby([w_plot])
    ax = fig2.add_subplot(4,4,i+1)
    ax.set_title(f'{w_plot} = {w}')
    ax.set_xscale('log')
    for x,w2 in enumerate(subg.groups):
        sg = subg.get_group(w2)
        if i == 0:
            plt.plot(sg['speed'], sg[var], label = f'{w_plot} = {w2}')
            #sg.plot( x = 'speed', y = 'ant_RG_drive_space', label =f'wBA = {w2}', ax = ax)
        else:
            plt.plot(sg['speed'], sg[var])
            #sg.plot( x = 'speed', y = 'ant_RG_drive_space', ax = ax)

    fig2.legend()




# # multiple heatmaps, keeping w- fixed and changing w+
# dfgrouped = df.groupby(['wBA'])

# fig = plt.figure()
# fig.suptitle('Heatmaps for wAB, keeping wBA fixed')

# for i,w in enumerate(dfgrouped.groups):
#     ax = fig.add_subplot(4,4,i+1)
#     ax.set_title(f'wBA = {w}')
#     g = dfgrouped.get_group(w)

#     # plot heatmap for beta
#     g_heatmap = g.pivot('wAB','speed','ant_RG_drive_space')
#     sns.heatmap(g_heatmap, ax = ax)


#     subg = g.groupby(['wAB'])
#     ax = fig2.add_subplot(4,4,i+1)
#     ax.set_title(f'wBA = {w}')
#     for x,w2 in enumerate(subg.groups):
#         sg = subg.get_group(w2)
#         if x == 0:
#             sg.plot( x = 'speed', y = 'ant_RG_drive_space', label =f'wAB = {w2}', ax = ax)
#         else:
#             sg.plot( x = 'speed', y = 'ant_RG_drive_space', ax = ax)

#     fig2.legend()

plt.show()



# same same for onset 


x = 0

# ax[1].set_title('Anticipation for different weights')
# wBAdf = df.groupby(['wBA'])
# for w in wBAdf.groups:
#     g = wBAdf.get_group(w)
#     g.plot( x = 'speed', y = 'ant_space', label = w, ax = ax[1])

# ax[1].legend()

# ax[2].set_title('Anticipation for different speeds')
# sdf = df.groupby(['speed'])
# for s in sdf.groups:
#     g = sdf.get_group(s)
#     g.plot( x = 'wBA', y = 'ant_space', label = s, ax = ax[2])

# # plot a against weight for many speeds
# x = 0
# plt.show()
# plot a against speed for many weights