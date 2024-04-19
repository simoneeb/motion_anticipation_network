import numpy as np
import scipy


def bar(t, xc, b = 0.160, v = 1):

    if xc >=-b+v*t and xc <=b+v*t :
        return 1
    else: 
        return 0
    

def measure_onset_anticipation(sim):
    
    
    onset_sim =np.argmax(sim >=1)
    #onset_ref =np.argmax(ref >=1)

    return onset_sim


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def DOG(x, mu, sig_c,sig_s,w):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_c, 2.))) - w*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig_s, 2.)))




def GainF_B(A):

    if A < 0 :
        return 0 
    else:
        return 1/(1+A**6)
    


def GainF_G(A):

    if A < 0 :
        return 0 
    else:
        return 1/(1+A**1)
    


def check_symmetry(A, tol=1e-8):
    return np.all(np.abs(np.abs(A)-np.abs(A.T)) < tol)