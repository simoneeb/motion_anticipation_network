import numpy as np



def N(V,params,celltype):

    """
    piecewise-linear nonlinearity

    """

    slope = params.get(f'slope_{celltype}',1)
    threshold = params.get(f'threshold_{celltype}',0)

    
    if V <= threshold:
        return 0
    else:
        return (V-threshold)*slope



def sig(x,  params):

    """
    sigmoidal nonlinearity
    
    """

    slope = params.get('slope',1)
    threshold = params.get('threshold',0)
    max_val = params.get('max_val',0)


    return max_val/ (1 + np.exp(-slope * (x - threshold)))