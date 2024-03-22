
import pickle
import numpy as np 
import matplotlib.pyplot as plt

from utils import GainF
from nonlinearities import N


# load simulation for bipolar pooling

net_name = 'selma_net_bipolar_pooling'
params_name = 'initial'
stim_name = 'smooth_3.0'
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}/{params_name}/{stim_name}'
save = True


# load params
with open(f'{fp}/out', 'rb') as handle:
    outp = pickle.load(handle)

with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)



dt = params['dt']
# measure peak timepoint for all BCells

tB_pooling = []
for Vi in outp['res'][0]['X']:
    tB_pooling.append(Vi.argmax()*dt)
tB_pooling = np.array(tB_pooling)

pooling_ref = outp['res'][0]['X'][50]
# load simulation with gain control 

net_name = 'selma_net_bipolar_pooling_gaincontrol_tauActB0.2'
params_name = 'initial'
stim_name = 'smooth_3.0'
fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}/{params_name}/{stim_name}'
save = True


# load params
with open(f'{fp}/out', 'rb') as handle:
    out = pickle.load(handle)

with open(f'{fp}/params', 'rb') as handle:
    params = pickle.load(handle)


# measure peak timepoint for all BCells

tB_GC = []
maxB_GC = []
for i,Vi in enumerate(out['res'][0]['X']):

    Ai = out['res'][0]['A'][i]
    Ni = np.array([N(v,params,'BC') for v in Vi]) 
    Gi = np.array([GainF(a) for a in Ai])

    Ri = Ni*Gi



    tB_GC.append(Ri.argmax()*dt)
    maxB_GC.append(Ri.max())
tB_GC = np.array(tB_GC)

x = 0

# calculate delta B Gain Control


deltaB_GC = np.mean(tB_pooling[50] - tB_GC[50])
maxB_GC = maxB_GC[50]

#w = np.concatenate((np.arange(1,100,1),np.arange(100,300,10)))
w = np.arange(0,160,10)
deltaB_I_by_weight = []
maxB_I_by_weight = []


fig,ax = plt.subplots(len(w),1,sharex= True)

for g,wi in enumerate(w):

    # load simulation with lateral inhibition
    net_name = 'selma_net_bipolar_pooling_lateral'
    params_name = 'initial'
    stim_name = 'smooth_3.0'
    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}/{params_name}/{stim_name}/w_explore/{wi}'
    save = True


    # load params
    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)


    # measure peak timepoint for all BCells

    # same for lateral inhibition 

    tB_I = []
    maxB_I = []
    for i,Vi in enumerate(out['res'][0]['X']):

        Ai = out['res'][0]['A'][i]
        Ni = np.array([N(v,params,'BC') for v in Vi]) 
        Gi = np.array([GainF(a) for a in Ai])

        Ri = Ni*Gi
        

        if i == 50 : 
            item = ax[g].plot(Ri, label = f'lateral inhibition w = {wi}')
            ax[g].axvline(Ri.argmax(), color = item[0].get_color())

            item = ax[g].plot(pooling_ref, label = 'pooling')
            ax[g].axvline(pooling_ref.argmax(),  color = item[0].get_color())



        tB_I.append(Ri.argmax()*dt)
        maxB_I.append(Ri.max())
    tB_I = np.array(tB_I)
    maxB_I = np.array(maxB_I)



    # calculate delta B Gain Control


    deltaB_I = np.mean(tB_pooling[50] - tB_I[50])
    maxB_I = maxB_I[50]
    deltaB_I_by_weight.append(deltaB_I)
    maxB_I_by_weight.append(maxB_I)




deltaB_I_GC_by_weight = []
maxB_I_GC_by_weight = []


for g,wi in enumerate(w):

    # load simulation with lateral inhibition
    net_name = 'selma_net_bipolar_pooling_lateral_gaincontrol_tauActB0.2'
    params_name = 'initial'
    stim_name = 'smooth_3.0'
    fp = f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/{net_name}/{params_name}/{stim_name}/w_explore/{wi}'
    save = True


    # load params
    with open(f'{fp}/out', 'rb') as handle:
        out = pickle.load(handle)

    with open(f'{fp}/params', 'rb') as handle:
        params = pickle.load(handle)


    # measure peak timepoint for all BCells

    # same for lateral inhibition 

    tB_I_GC = []
    maxB_I_GC = []
    for i,Vi in enumerate(out['res'][0]['X']):

        Ai = out['res'][0]['A'][i]
        Ni = np.array([N(v,params,'BC') for v in Vi]) 
        Gi = np.array([GainF(a) for a in Ai])

        Ri = Ni*Gi
        

        if i == 50 : 
            item = ax[g].plot(Ri, label = f'lateral inhibition w = {wi}')
            ax[g].axvline(Ri.argmax(), color = item[0].get_color())


        tB_I_GC.append(Ri.argmax()*dt)
        maxB_I_GC.append(Ri.max())
    tB_I_GC = np.array(tB_I_GC)
    maxB_I_GC = np.array(maxB_I_GC)



    # calculate delta B Gain Control


    deltaB_I_GC = np.mean(tB_pooling[50] - tB_I_GC[50])
    maxB_I_GC = maxB_I_GC[50]
    deltaB_I_GC_by_weight.append(deltaB_I_GC)
    maxB_I_GC_by_weight.append(maxB_I_GC)





x = 0


#plot anticipation vs weight

fog = plt.figure()

plt.axhline(deltaB_GC, label = 'Gain control alone', color = (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0))

plt.plot(w,deltaB_I_by_weight, color = 'grey')
plt.scatter(w,deltaB_I_by_weight, color = (0.403921568627451, 0.0, 0.05098039215686274, 1.0), marker = 'v', label = 'Without gain control')

plt.plot(w,deltaB_I_GC_by_weight, color = 'grey')
plt.scatter(w,deltaB_I_GC_by_weight, facecolor ='none', marker = 's', edgecolor = (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0), label = 'With gain control')


plt.ylabel('anticipation [s]')
plt.xlabel('w [Hz]')

plt.legend()

fog.savefig(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/figure7A_tauActB0.22.png')


#plot anticipation vs weight

fog = plt.figure()

plt.axhline(maxB_GC, label = 'Gain control alone')

plt.plot(w,maxB_I_by_weight,  color = (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0))
plt.scatter(w,maxB_I_by_weight, color = (0.403921568627451, 0.0, 0.05098039215686274, 1.0), marker = 'v', label = 'Without gain control')

plt.plot(w,maxB_I_GC_by_weight, color = 'grey')
plt.scatter(w,maxB_I_GC_by_weight, facecolor ='none', marker = 's', edgecolor = (0.03137254901960784, 0.18823529411764706, 0.4196078431372549, 1.0),label = 'With gain control')


plt.ylabel('maximum bipolar voltage [V]')
plt.xlabel('w [Hz]')

plt.legend()

fog.savefig(f'/user/sebert/home/Documents/Simulations/motion/anticipation_1D/selma/figure7B_tauActB0.22.png')



