
import os
import numpy as np
from joblib import Parallel,delayed
from run_Reciporcal import run_Reciporcal
from model.params_FB import make_params
import time
import pandas as pd
import pickle
import sys


from utils import measure_onset_anticipation

# def measure_onset_anticipation(sim):

#     onset_sim =np.argmax(sim >=1)
#     #onset_ref =np.argmax(ref >=1)

#     return onset_sim
#     #return onset_ref - onset_sim

net_name = f'Reciporcal_mono_linear_heavy_maxan_equalweight/noGCGainControl'
net_name = f'fb_thesis_linear_nonorm'

stim_type = 'smooth'

# loop over parameter
n_params = 20
vals1 = np.linspace(0.09,0.31,n_params)
vals2 = np.linspace(1,101,n_params) 
# vals2 =[0.01,0.3]
# vals1 =[1,10]
speeds = [0.1,0.5,1.0,2.0]
#speeds = [0.1,0.4,0.7,1.0,2.0]

# speeds = [0.5,0.8]
nb_jobs = len(vals1)*len(vals2)*len(speeds)
dur = (nb_jobs*11.5)/60
print(f'nb jobs :  {nb_jobs}, takes {dur} mins')

# load ref for motion onset

# refdict = {}

# for si in speeds:
#     fp = f'~/Documents/Simulations/motion_anticipation_network/Loops/{net_name}/wAB/wAB_0.0/smooth_{si}'
#     #fp = f'~/Documents/Simulations/motion_anticipation_network/Loops/{net_name}/wGA/wGA_0.0/smooth_{si}'

#     with open(f'{fp}/out', 'rb') as handle:
#         out = pickle.load(handle)
#     with open(f'{fp}/params', 'rb') as handle:
#         params = pickle.load(handle)

#     dt = params['dt']

#     refdict[f'{si}'] = {}
#     refdict[f'{si}']['RG'] = out['RG']
#     refdict[f'{si}']['RB'] = out['RB'][50]
#     refdict[f'{si}']['max_tp_RG'] = np.argmax(out['RG'])*dt
#     refdict[f'{si}']['max_tp_RB'] = np.argmax(out['RB'][50])*dt

df = pd.DataFrame(columns=['wTOT','tauTOT','wBA', 'wAB', 'tauA', 'tauB','mu', 'speed', 'peak_RG','peak_RB', 'peak_drive', 'tp_rf_GC_mid', 'peak_RG_pooling', 'peak_RB_pooling', 'onset_RB', 'onset_RG'])
# df = pd.DataFrame(columns=['wAB','wBA', 'speed', 'ant_space', 'ant_time', 'ant_space_drive', 'ant_time_drive', 'onset_shift'])
dfresRG = pd.DataFrame()
dfresRB = pd.DataFrame()
grid = np.array(np.meshgrid(vals1,vals2,speeds)).T.reshape(-1,3)


#TODO add output to dataframe durinng the parallel run to avoid blocking memory
def run(val1,val2,si):

    params = make_params()

    tauB = params['tauB']
    wBA =  46.0
    # tauTOT = np.round(val1,2)
    # wTOT = np.round(val2,2)
    
    # tauA = 1/(tauTOT + 1/tauB)
    # wBA = wTOT/wAB

    tauA = np.round(val1,2)
    wAB = np.round(val2,2)
    
    tauTOT = 1/tauA - 1/tauB
    wTOT = wBA*wAB




    mu = wTOT*(tauTOT**2)

    # print(tauA)
    # print(wBA)
    params = make_params(param_names = ['speed','wAB','tauA','wGA','wBA'], param_vals=[si,wAB,tauA,0,1.0])

    [peak_RG,peak_RB,peak_drive,tp_rf_GC_mid,onset_RG,onset_RB,RG,RB,VG] = run_Reciporcal(params = params)   


    # [ant_space,ant_time,ant_space_drive,ant_time_drive,RG] = run_Reciporcal(params = params)   
    # onset_RG = measure_onset_anticipation(RG)
    # onset_RB = measure_onset_anticipation(RB)
    # peak_RG_pooling = refdict[f'{si}']['max_tp_RG']
    # peak_RB_pooling = refdict[f'{si}']['max_tp_RB']

    RG = RG[0::10]
    RB = RB[0::10]

    data = {'wAB': params['wAB'],
            'tauB': params['tauB'],
            'tauTOT': tauTOT,
            'wTOT': wTOT,
            'tauA': tauA,
            'wBA': wBA,
            'mu': mu,
            'speed' : si,
            'peak_RG' : peak_RG,
            'peak_RB' : peak_RB,
            'peak_drive' : peak_drive,
            # 'peak_RG_pooling' : peak_RG_pooling,
            # 'peak_RB_pooling' : peak_RB_pooling,
            'tp_rf_GC_mid' : tp_rf_GC_mid,
            'onset_RG' : onset_RG,
            'onset_RB' : onset_RB}
    
    
  
    
    return [data,RG,RB,params]



start = time.time()

X = Parallel(n_jobs = 20, verbose=10)(delayed(run)(i[0],i[1],i[2]) for i in grid)

print(sys.getsizeof(X))

for i,xi in enumerate(X):
    data = xi[0]
    speed = data['speed']
    # data['peak_RG_pooling'] =  refdict[f'{speed}']['max_tp_RG']
    # data['peak_RB_pooling'] =  refdict[f'{speed}']['max_tp_RB']

    # data['onset_RG_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RG'])
    # data['onset_RB_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RB'])

    df = df._append(data, ignore_index = True)
    # data = pd.DataFrame(data)
    # df = pd.concat([df,data], ignore_index = True, axis =1)
    

    RG = xi[1]
    RB = xi[2]
    dfnRG = pd.DataFrame({ f'{i}' : RG})
    dfnRB = pd.DataFrame({ f'{i}' : RB})
    dfresRG = pd.concat([dfresRG,dfnRG], ignore_index=True, axis=1)
    dfresRB = pd.concat([dfresRB,dfnRB], ignore_index=True, axis=1)




print(sys.getsizeof(df))


home = os.path.expanduser("~")
filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

df.to_csv(f'{filepath}/anticipation_data_mu.csv')
dfresRG.to_csv(f'{filepath}/responses_RG_mu.csv')
dfresRB.to_csv(f'{filepath}/responses_RB_mu.csv')

stop = time.time()
params = X[-1][-1]
with open(f'{filepath}/params_grid_mu', 'wb') as handle:
            pickle.dump(params, handle)


print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))

