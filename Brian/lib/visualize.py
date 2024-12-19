import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def plot_grid(output,params):
   
    [stim,sigB,sigA,sigG] =  output

    N,T = stim.shape


    timeticks = np.arange(0,T)*params['dt']
    xticks = np.arange(0,T)

    spaceticks = np.arange(0,N)*params['spacing']
    yticks = np.arange(0,N)

    fig = plt.figure(figsize = (10,10))
    gs = fig.add_gridspec(4,1)
    fig.subplots_adjust(hspace = 0.6)
    

    ax = fig.add_subplot(gs[0,0])
    im = ax.imshow(stim, cmap = 'Greys')
    ax.set_title('stimulus', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('x [mm] ')
    ax.set_xticks(xticks[::500],timeticks[::500])
    ax.set_yticks(yticks[::200],spaceticks[::200])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')



    ax = fig.add_subplot(gs[1,0])
    im =ax.imshow(sigB, cmap = "Blues")
    ax.set_title('$V_{B}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('x [mm] ')
    ax.set_xticks(xticks[::500],timeticks[::500])
    ax.set_yticks(yticks[::200],spaceticks[::200])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')



    ax = fig.add_subplot(gs[2,0])
    im =ax.imshow(sigA, cmap = 'Reds')
    ax.set_title(r'$V_{A}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('x [mm] ')
    ax.set_xticks(xticks[::500],timeticks[::500])
    ax.set_yticks(yticks[::200],spaceticks[::200])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')



    ax = fig.add_subplot(gs[3,0])
    im =ax.imshow(sigG, cmap = 'Greys')
    ax.set_title(r'$V_{G}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('x [mm] ')
    ax.set_xticks(xticks[::500],timeticks[::500])
    ax.set_yticks(yticks[::200],spaceticks[::200])


    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


    return fig

def plot_timetrace(output,params):
   
    [stim,sigB,sigA,sigG] =  output
    N,T = stim.shape


    timeticks = np.arange(0,T)*params['dt']
    xticks = np.arange(0,T)

    spaceticks = np.arange(0,N)*params['spacing']
    yticks = np.arange(0,N)

    fig = plt.figure(figsize = (10,10))
    gs = fig.add_gridspec(4,1)
    fig.subplots_adjust(hspace = 0.6)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(timeticks,stim[206,:], color = 'grey')
    ax.set_title('stimulus', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel(' ')
    ax.set_xticks(timeticks[::500],timeticks[::500])
    ax.set_yticks([])



    ax = fig.add_subplot(gs[1,0])
    ax.plot(timeticks,sigB[206,:], color = 'royalblue')
    ax.set_title('$V_{B}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('V [mV] ')
    ax.set_xticks(timeticks[::500],timeticks[::500])



    ax = fig.add_subplot(gs[2,0])
    ax.plot(timeticks,sigA[206-1,:], color = 'red')
    ax.set_title(r'$V_{A}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('V [mV] ')
    ax.set_xticks(timeticks[::500],timeticks[::500])



    ax = fig.add_subplot(gs[3,0])
    ax.plot(timeticks,sigG[206,:],color = 'black')
    ax.set_title(r'$V_{G}$', loc = 'left')
    ax.set_xlabel('time [s] ')
    ax.set_ylabel('V [mV] ')
    ax.set_xticks(timeticks[::500],timeticks[::500])


    return fig




def plot_range(outputs,params,range):

    [stims,sigBs,steadies] =  outputs 


    fig = plt.figure(figsize = (10,3))
    fig.subplots_adjust(wspace = 0.6)
    gs = fig.add_gridspec(1,3)


    ax = fig.add_subplot(gs[0,0])
    ax.set_title('stimulus', loc = 'left')
    ax.set_xlabel('time [s]')

    for ci, c in enumerate(range):

        N,T = stims[ci].shape
        timeticks = np.arange(0,T)*params['dt']

        ax.plot(timeticks,stims[ci][206,:], label = f'{c}')


    ax = fig.add_subplot(gs[0,1])
    ax.set_title('$V_{B}$', loc = 'left')
    ax.set_xlabel('time [s]')
    ax.set_ylabel(' [mV]')

    for ci, c in enumerate(range):
        ax.plot(timeticks,sigBs[ci][206,:])

    ax = fig.add_subplot(gs[0,2])
    ax.set_title('$V_{B}*$', loc = 'left')
    ax.set_xlabel('contrast')

    for ci, c in enumerate(range):
        ax.scatter(c,steadies[ci])

    return fig