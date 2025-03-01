
import os
import numpy as np
from joblib import Parallel,delayed
from run_Reciporcal import run_Reciporcal
from model.params_FB import make_params
import time
import pandas as pd
import pickle
from utils import measure_onset_anticipation

def calc_n_f(krel,krec,bet,frec):
    return krec/(krec+krel*bet*frec)



#loop over parameter
krel = 1.
krec = .5
# range of beta
#speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,2.0]
#speeds = [0.1,0.4,0.7,1.0,2.0]


#loop over tau-n
#vals = np.arange(0.0001,0.001,.0002)
vals = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
vals = vals[::2]
#vals = [0.001,0.6,1]
# vals = [0.0,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.012,0.015,0.017,0.02,0.025,0.03,0.035,0.04,0.045,0.05,0.055,0.6,0.065,0.07,0.075,0.08,0.085,0.09,0.095,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]
# vals = [0.0,0.001,0.002]

#vals = [0.01,0.05,0.1,0.5,0.6,1,2,5,10]
net_name = f'Reciporcal_mono_linear_heavy_maxan_equalweight/noGCGainControl'
net_name = f'fb_thesis_linear'

stim_type = 'smooth'

#loop over parameter

# vals2 =[0.01,0.3]
# vals1 =[1,10]
speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,2.0]
speeds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]

#speeds = [0.2,0.5,1.0]

# speeds = [0.5,0.8]
nb_jobs = len(vals)**2*len(speeds)
print(f'nb jobs :  {nb_jobs}')
# load ref for motion onset

# refdict = {}

# for si in speeds:
#     fp = f'~/Documents/Simulations/motion_anticipation_network/Loops/fb_thesis_linear/wAB/wAB_0.0/smooth_{si}'

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

df = pd.DataFrame(columns=['wBA', 'wAB', 'tauA', 'tauB', 'speed', 'peak_RG','peak_RB', 'peak_drive', 'tp_rf_GC_mid', 'peak_RG_pooling', 'peak_RB_pooling', 'onset_RB', 'onset_RG','taunB'])
# df = pd.DataFrame(columns=['wAB','wBA', 'speed', 'ant_space', 'ant_time', 'ant_space_drive', 'ant_time_drive', 'onset_shift'])
dfresRG = pd.DataFrame()
dfresRB = pd.DataFrame()
grid = np.array(np.meshgrid(vals,vals,speeds)).T.reshape(-1,3)


def run(val1,val2,si):

    betaB = val1
    kratioB = krec/krel
    taunB = 1/(krec+krel*betaB)
    ncalcB = calc_n_f(krec,krel,betaB,si)


    betaA = val2
    kratioA = krec/krel
    taunA = 1/(krec+krel*betaA)
    ncalcA = calc_n_f(krec,krel,betaA,si)


    params = make_params(param_names = ['speed','krecB','krelB','betaB','krecA','krelA','betaA','wGA','wBA','plastic_to_G','plastic_to_A'], param_vals=[si,krec,krel,betaB,krec,krel,betaA,0.0,46.0,1,0])

    [peak_RG,peak_RB,peak_drive,tp_rf_GC_mid,onset_RG,onset_RB,RG,RB,nmin_B, nmin_A] = run_Reciporcal(params = params, measure_n=True)   

    # print(f'krel : {krel}')
    # print(f'krec : {krec}')
    # print(f'beta : {beta}')


    RG = RG[0::10]
    RB = RB[0::10]

    data = {'krecB' : krec,
            'krelB' : krel,
            'kratioB' : kratioB,
            'betaB' : betaB,
            'neqB' : ncalcB,
            'taunB' : taunB,
            'nminB' : nmin_B,
            'krecA' : krec,
            'krelA' : krel,
            'kratioA' : kratioA,
            'betaA' : betaA,
            'neqA' : ncalcA,
            'taunA' : taunA,
            'nminA' : nmin_A,
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

for i,xi in enumerate(X):
    data = xi[0]
    speed = data['speed']
    # data['peak_RG_pooling'] =  refdict[f'{speed}']['max_tp_RG']
    # data['peak_RB_pooling'] =  refdict[f'{speed}']['max_tp_RB']

    # data['onset_RG_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RG'])
    # data['onset_RB_pooling'] =  measure_onset_anticipation(refdict[f'{speed}']['RB'])

    df = df._append(data, ignore_index = True)
    



    RG = xi[1]
    RB = xi[2]
    dfnRG = pd.DataFrame({ f'{i}' : RG})
    dfnRB = pd.DataFrame({ f'{i}' : RB})
    dfresRG = pd.concat([dfresRG,dfnRG], ignore_index=True, axis=1)
    dfresRB = pd.concat([dfresRB,dfnRB], ignore_index=True, axis=1)


home = os.path.expanduser("~")
filepath = f'{home}/Documents/Simulations/motion_anticipation_network/{net_name}'
if not os.path.isdir(filepath):
    os.makedirs(filepath)

df.to_csv(f'{filepath}/anticipation_data_tauNB_tauNA_in.csv')
dfresRG.to_csv(f'{filepath}/responses_RG_tauNB_tauNA_in.csv')
dfresRB.to_csv(f'{filepath}/responses_RB_tauNB_tauNA_in.csv')

stop = time.time()
params = X[-1][-1]
with open(f'{filepath}/params_grid_tauNB_tauNA_in', 'wb') as handle:
            pickle.dump(params, handle)



print('Elapsed time for the entire processing: {:.2f} s'
      .format(stop - start))