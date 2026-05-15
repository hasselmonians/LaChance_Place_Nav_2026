# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:11:45 2026

@author: plachanc
"""

import os
import copy
import numpy as np
import pandas as pd
import pickle
from scipy import ndimage
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import seaborn as sns

framerate = 30.

def polar_dot_plot(angles,savefile=None,labels=None):
    ''' make a polar dot plot '''
    
    shift_bins = 60.
    
    max_lim = 4.82279776210871
        
    angles = np.deg2rad((angles+(360./shift_bins)/2.)%360)
    
    circ_rad = 1.
    initial_spacing = .07
    dot_spacing = .1
    
    circ = lambda r : (r*np.cos(np.linspace(0,2.*np.pi,360)), r*np.sin(np.linspace(0,2.*np.pi,360)))
    pol2cart = lambda pol : (pol[0]*np.cos(pol[1]),pol[0]*np.sin(pol[1]))
    circ_x,circ_y=circ(circ_rad)
    
    bins = np.linspace(0,2.*np.pi,int(shift_bins)+1,endpoint=True)
    
    palette = ['green','blue','red','orange','magenta','gray','skyblue','yellow','aqua','pink','purple','cyan','black','white','indigo','violet','crimson']


    cell_counts = np.zeros(0)
    cell_hist_bins = np.zeros(0)
    animalnums = []
    colors = []
    rxs = []
    rys = []
    rs = []
    r_colors = []
    bin_counts = np.tile(np.arange(int(shift_bins)),len(np.unique(labels)))
    unique_labels = np.unique(labels)
    for l in range(len(unique_labels)):
        label = unique_labels[l]
        label_angles = angles[labels==label]
        c,h = np.histogram(label_angles,bins=bins)
        cell_counts = np.concatenate((cell_counts,c))
        cell_hist_bins = np.concatenate((cell_hist_bins,bins[:int(shift_bins)]))
        for i in range(len(c)):
            animalnums.append(label)
            colors.append(palette[l])
        
        mr = np.nansum(c*np.exp(1j*np.linspace(0,2.*np.pi,int(shift_bins),endpoint=False)))/np.nansum(c)
        mra = np.arctan2(np.imag(mr),np.real(mr))
        mrl = np.abs(mr)
        
        rx,ry = pol2cart((mrl,mra))
        rxs.append(rx)
        rys.append(ry)
        rs.append(mrl)
        r_colors.append(palette[l])
        
    xs = []
    ys = []
    animal_labels = []
    color_labels = []
    counts = np.zeros(int(shift_bins))
    spacing = np.zeros(int(shift_bins))
    for i in range(len(cell_counts)):
        for j in range(int(cell_counts[i])):
            r = circ_rad + initial_spacing + spacing[int(bin_counts[i])]
            theta = cell_hist_bins[i]
            x,y = pol2cart((r,theta))
            xs.append(x)
            ys.append(y)
            animal_labels.append(animalnums[i])
            color_labels.append(colors[i])
            spacing[int(bin_counts[i])] += dot_spacing
            counts[int(bin_counts[i])] += 1.
                
                
    rx,ry = pol2cart((mrl,mra))
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
#    ax=plt.gca()
    ax.plot(circ_x,circ_y,'k-')
    if labels is None:
        ax.plot(xs,ys,'ko',clip_on=False)
        ax.arrow(0,0,rx,ry,head_width=0.12,head_length=0.13,fc='k',ec='k',linewidth=2,length_includes_head=True,clip_on=False)
    else:
        for animal in unique_labels:
            ix = np.where(np.array(animal_labels)==animal)[0]
            ax.scatter(np.array(xs)[ix],np.array(ys)[ix],c=np.array(color_labels)[ix],label=animal,s=76,clip_on=False)
            
            
    longest_axis = np.max([ax.get_xlim()[1]-ax.get_xlim()[0],ax.get_ylim()[1]-ax.get_ylim()[0]])
    height = 10. * longest_axis/max_lim
    width = 10. * longest_axis/max_lim
    
    fig.set_figwidth(width)
    fig.set_figheight(height)

    for i in range(len(rxs)):
        ax.arrow(0,0,rxs[i],rys[i],head_width=0.14,head_length=0.13,fc=r_colors[i],ec=r_colors[i],linewidth=2,length_includes_head=True)
        # ax.text(.3,.6-.17*float(i),'r=%s' % str(round(rs[i],2)),transform=ax.transAxes,color=r_colors[i],fontsize=20) 
        
    ax.plot([-1,-.95],[0,0],'k-')
    ax.text(-.81,-.01,'180',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([.95,1],[0,0],'k-')
    ax.text(.88,-.01,'0',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([0,0],[-1,-.95],'k-')
    ax.text(0,-.88,'-90',horizontalalignment='center',verticalalignment='center',fontsize=18)
    ax.plot([0,0],[.95,1],'k-')
    ax.text(0,.86,'90',horizontalalignment='center',verticalalignment='center',fontsize=18)


    plt.axis('equal')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.rc('axes.spines', **{'bottom':False, 'left':False, 'right':False, 'top':False})
    
    if savefile is not None:
        dest_folder = os.path.dirname(savefile)
        if not os.path.isdir(dest_folder):
            os.makedirs(dest_folder)
            
        fig.patch.set_visible(False)
        ax.axis('off')
        
        fig.savefig(savefile)
        plt.close()
        
    else:
        plt.show()
        
        


def make_figures():
    
    overdir = os.getcwd()
    
    data_dir = overdir + '/data'
    fig_dir = overdir + '/figures'
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
        
    ''' Figure 1 '''
    
    fig1_dir = fig_dir + '/fig1'
    if not os.path.exists(fig1_dir):
        os.makedirs(fig1_dir)
    
    ''' panel 1A '''
    
    with open(data_dir + '/tetrode/panel_1AG.pickle','rb') as f:
        behavior_1A = pickle.load(f)
    
    #closest video frame to timestamps for entry/exit 11 (one used in Fig. 1A)
    entry_diffs = abs(behavior_1A['timestamps'] - behavior_1A['entry_ts'][10])
    entry_frame_ind = np.argmin(entry_diffs)
    exit_diffs = abs(behavior_1A['timestamps'] - behavior_1A['exit_ts'][10])
    exit_frame_ind = np.argmin(exit_diffs)
    
    #define entry and exit path lengths to plot
    entry_length = 1 #seconds
    exit_length = 3. #seconds
    
    #plot gray path
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(behavior_1A['pos_x'],behavior_1A['pos_y'],'gray',alpha=.6)

    #plot entry path
    entry_x = behavior_1A['pos_x'][(entry_frame_ind - int(entry_length * framerate)) : entry_frame_ind + 1]
    entry_y = behavior_1A['pos_y'][(entry_frame_ind - int(entry_length * framerate)) : entry_frame_ind + 1]
    entry_hd = behavior_1A['angles'][entry_frame_ind]
    ax.plot(entry_x,entry_y,'green',linewidth=3)
    ax.quiver(entry_x[0],entry_y[0],np.cos(np.deg2rad(entry_hd)),np.sin(np.deg2rad(entry_hd)),headwidth=8,headlength=14,headaxislength=14,clip_on=False,pivot='mid',zorder=1000)

    #plot exit path
    exit_x = behavior_1A['pos_x'][exit_frame_ind : (exit_frame_ind + int(exit_length * framerate)) + 1]
    exit_y = behavior_1A['pos_y'][exit_frame_ind : (exit_frame_ind + int(exit_length * framerate)) + 1]
    exit_hd =  behavior_1A['angles'][exit_frame_ind + int(exit_length * framerate)]
    ax.plot(exit_x,exit_y,'#C55A11',linewidth=3)
    ax.quiver(exit_x[-1],exit_y[-1],np.cos(np.deg2rad(exit_hd)),np.sin(np.deg2rad(exit_hd)),headwidth=8,headlength=14,headaxislength=14,clip_on=False,pivot='mid',zorder=1000)
    
    #plot within-zone path
    between_x = behavior_1A['pos_x'][entry_frame_ind : exit_frame_ind+1]
    between_y = behavior_1A['pos_y'][entry_frame_ind : exit_frame_ind+1]
    ax.plot(between_x,between_y,'gray',linewidth=3)
    
    #plot zone
    zone_coords = behavior_1A['zone_coords']
    ax.plot([zone_coords['x1'],zone_coords['x2'],zone_coords['x2'],zone_coords['x1'],zone_coords['x1']],
            [zone_coords['y1'],zone_coords['y1'],zone_coords['y2'],zone_coords['y2'],zone_coords['y1']],'k-',linewidth=3,alpha=.9,zorder=1000)
    
    ax.axis('equal')
    ax.axis('off')
    fig.savefig(fig1_dir + '/panel_1A.png', dpi=400)
    plt.close()
    
    
    ''' panel 1B '''
    
    #load data
    behavior_data = pd.read_csv(data_dir + '/tetrode/behavior_csv.csv')
    
    #get baseline post-training sessions only
    baseline_behavior = behavior_data[(behavior_data['trial']=='1.2m cue_south s1')|(behavior_data['session']=='training_sessions')]
    
    #assign colors to animal IDs
    animal_ids = np.array(baseline_behavior['animal'])
    colors = np.full_like(animal_ids,'gray',dtype='<U8')
    colors[animal_ids=='EG2'] = 'blue'
    colors[animal_ids=='EG3'] = 'green'
    colors[animal_ids=='SS1'] = 'red'
    colors[animal_ids=='SS2'] = 'purple'
    
    #plot the learning curve for rewards/minute
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(baseline_behavior['exposure'],baseline_behavior['rewards']/baseline_behavior['session_length_minutes'],c=colors,clip_on=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel('Rewards per minute')
    ax.set_xlabel('Day')
    ax.set_ylim([0,None])
    ax.set_xlim([0,65])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.8*(x1-x0)/(y1-y0))
    fig.savefig(fig1_dir + '/panel_1B.png', dpi=400)
    plt.close()
    
    ''' panel 1C '''
    
    #plot the learning curve for rewards/meter
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(baseline_behavior['exposure'],baseline_behavior['rewards']/baseline_behavior['path_length_meters'],c=colors,clip_on=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel('Rewards per minute')
    ax.set_xlabel('Day')
    ax.set_ylim([0,None])
    ax.set_xlim([0,65])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.8*(x1-x0)/(y1-y0))
    fig.savefig(fig1_dir + '/panel_1C.png', dpi=400)
    plt.close()


    ''' panel 1D '''

    with open(data_dir + '/tetrode/panel_1D.pickle','rb') as f:
        behavior_1D = pickle.load(f)
    
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5

    #plot path, goal zone, and equivalent zones
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(behavior_1D['pos_x'],behavior_1D['pos_y'],'gray',alpha=.6)
    ax.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1],'r-',linewidth=3)
    ax.plot([120.-y1,120.-y1,120-y2,120-y2,120-y1],[x1,x2,x2,x1,x1],'k-',linewidth=3)
    ax.plot([120.-x1,120.-x2,120.-x2,120.-x1,120.-x1],[120.-y1,120.-y1,120-y2,120-y2,120-y1],'k-',linewidth=3)
    ax.plot([y1,y1,y2,y2,y1],[120.-x1,120.-x2,120.-x2,120.-x1,120.-x1],'k-',linewidth=3)
    ax.axis('equal')
    ax.axis('off')
    fig.savefig(fig1_dir + '/panel_1D.png', dpi=400)
    plt.close()
    
    
    ''' panel 1E '''

    #entries per minute by zone
    zones = ['true','cw','opp','ccw']
    colors = ['red','gray','gray','gray']
    edgecolors = ['black','black','black','black']
    
    jitter = .1 * np.random.normal(size=len(baseline_behavior))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    count = 0
    for i in range(len(zones)):
        zone=zones[i]
        color=colors[i]
        mean_val = np.mean(baseline_behavior['%s_%s' % (zone,'entries_per_minute')])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,baseline_behavior['%s_%s' % (zone,'entries_per_minute')],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,3.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,3.5])
    ax.set_xticks([0,1,2,3],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    fig.savefig(fig1_dir + '/panel_1E.png',dpi=400)
    plt.close()
    
    
    ''' panel 1F '''
    
    #occupancy time by zone
    fig = plt.figure()
    ax = fig.add_subplot(111)
    count = 0
    for i in range(len(zones)):
        zone=zones[i]
        color=colors[i]
        mean_val = np.mean(baseline_behavior['%s_%s' % (zone,'occ')])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,baseline_behavior['%s_%s' % (zone,'occ')],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,3.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,3.5])
    ax.set_xticks([0,1,2,3],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    fig.savefig(fig1_dir + '/panel_1F.png',dpi=400)
    plt.close()
    
    ''' panel 1G '''
    
    with open(data_dir + '/tetrode/panel_1AG.pickle','rb') as f:
        behavior_1G = pickle.load(f)
    
    #create subplots
    fig, axes = plt.subplots(1,2)
    
    #separately plot zone entries and exits
    for path in ['arrival','departure']:
        
        if path == 'arrival':
            relevant_timestamps = behavior_1G['entry_ts']
        else:
            relevant_timestamps = behavior_1G['exit_ts']
        
        #assign timestamps to closest video frame
        frame_inds = []
        for i in relevant_timestamps:
            abs_diffs = abs(behavior_1G['timestamps'] - i)
            closest_ind = np.argmin(abs_diffs)
            frame_inds.append(closest_ind)
        
        #compute 1 second paths before entry / after exit, along with starting / ending HDs
        time_length = 1. #seconds
        paths = []
        start_hds = []
        end_hds = []
        for ind in frame_inds:
            if len(behavior_1G['pos_x'][(ind - int(time_length * framerate)):(ind + int(time_length * framerate))]) == int(2 * time_length * framerate):
                if path == 'arrival':
                    paths.append([behavior_1G['pos_x'][(ind - int(time_length * framerate)):ind],behavior_1G['pos_y'][(ind - int(time_length * framerate)):ind]])
                elif path == 'departure':
                    paths.append([behavior_1G['pos_x'][ind:(ind + int(time_length * framerate))],behavior_1G['pos_y'][ind:(ind + int(time_length * framerate))]])
                
                start_hds.append(behavior_1G['angles'][ind - int(time_length * framerate)])
                end_hds.append(behavior_1G['angles'][ind])
                    
        #plot the entry or exit paths
        if path == 'arrival':
            ax_ind = 0
        elif path == 'departure':
            ax_ind = 1
        axes[ax_ind].plot(behavior_1G['pos_x'],behavior_1G['pos_y'],'gray',alpha=.6)
        for i in range(len(paths)):
            a = paths[i]
            axes[ax_ind].plot(a[0],a[1])
            if path == 'arrival':
                axes[ax_ind].quiver(a[0][0],a[1][0],np.cos(np.deg2rad(start_hds[i])),np.sin(np.deg2rad(start_hds[i])),headwidth=8,headlength=14,headaxislength=14,clip_on=False,pivot='mid',zorder=1000)
            elif path == 'departure':
                axes[ax_ind].quiver(a[0][-1],a[1][-1],np.cos(np.deg2rad(end_hds[i])),np.sin(np.deg2rad(end_hds[i])),headwidth=8,headlength=14,headaxislength=14,clip_on=False,pivot='mid',zorder=1000)
        axes[ax_ind].plot([zone_coords['x1'],zone_coords['x2'],zone_coords['x2'],zone_coords['x1'],zone_coords['x1']],
                [zone_coords['y1'],zone_coords['y1'],zone_coords['y2'],zone_coords['y2'],zone_coords['y1']],'k-')
        axes[ax_ind].axis('equal')
        axes[ax_ind].axis('off')
        
        if path == 'arrival':
            axes[ax_ind].set_title('Approaches')
        elif path == 'departure':
            axes[ax_ind].set_title('Departures')

    fig.savefig(fig1_dir + '/panel_1G.png', dpi=400)


    ''' panel 1H '''
    
    with open(data_dir + '/tetrode/approach_depart_hds_speeds.pickle','rb') as f:
        zone_hds = pickle.load(f)
    
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': 'polar'})
    
    #histogram of approach HDs
    axes[0].yaxis.grid(False)
    axes[0].xaxis.grid(linewidth=2,color='k',zorder=100)
    axes[0].spines['polar'].set_visible(False)
    axes[0].hist(np.deg2rad(zone_hds['approach_end']),bins=np.linspace(0,2.*np.pi,30,endpoint=True),facecolor='green')
    axes[0].set_xticks([0,np.pi/2,np.pi,3.*np.pi/2],['E','N','W','S'])
    axes[0].set_yticklabels([])
    axes[0].set_title('Approach HDs')
    
    #histogram of departure HDs
    axes[1].yaxis.grid(False)
    axes[1].xaxis.grid(linewidth=2,color='k',zorder=100)
    axes[1].spines['polar'].set_visible(False)
    axes[1].hist(np.deg2rad(zone_hds['depart_start']),bins=np.linspace(0,2.*np.pi,30,endpoint=True),facecolor='#C55A11')
    axes[1].set_xticks([0,np.pi/2,np.pi,3.*np.pi/2],['E','N','W','S'])
    axes[1].set_yticklabels([])
    axes[1].set_title('Departure HDs')
    plt.tight_layout()
    
    fig.savefig(fig1_dir + '/panel_1H.png', dpi=400)
    plt.close()
    
    
    ''' Figure 2 '''

    fig2_dir = fig_dir + '/fig2'
    if not os.path.exists(fig2_dir):
        os.makedirs(fig2_dir)
        
    ''' panels 2A-D '''
    
    #not data-based
    
    ''' panel 2E '''

    #load data for example cells
    with open(data_dir + '/tetrode/panel_2E.pickle','rb') as f:
        data_2E = pickle.load(f)    
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_2E['cell_%i' % i]['tracking']
        cell_mvls = data_2E['cell_%i' % i]['mvls']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #MVL plots
        
        #scale zone coordinates to 20 x 20 grid
        zone_scale = 120./20.
        
        axes[1, i].imshow(cell_mvls['mvls'],vmin=0,vmax = np.max(cell_mvls['mvls']),cmap='viridis',origin='lower') 
        axes[1, i].scatter(cell_mvls['max_x'],cell_mvls['max_y'],marker='x',color='k',s=100,linewidth=1,clip_on=False,zorder=1000)
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig2_dir + '/panel_2E.png', dpi=400)
    plt.close()
    
    
    ''' panel 2F '''
    
    #load all MVL data
    with open(data_dir + '/tetrode/cb_baseline_mvls.pickle','rb') as f:
        baseline_mvls = pickle.load(f)
        
    #consolidate MVL maps and MVLmax locations across all pure CB cells
    all_mvls = []
    max_x = []
    max_y = []
    
    for key in baseline_mvls.keys():
        all_mvls.append(baseline_mvls[key]['resampled_mvls'])
        max_x.append(baseline_mvls[key]['resampled_max_x'])
        max_y.append(baseline_mvls[key]['resampled_max_y'])

    #make plots
    fig, axes = plt.subplots(1, 2)
    counts, xedges, yedges, im = axes[0].hist2d(max_x,max_y,bins=np.arange(0,21),cmap='viridis')
    axes[0].plot([x1/zone_scale,x2/zone_scale,x2/zone_scale,x1/zone_scale,x1/zone_scale],
            [y1/zone_scale,y1/zone_scale,y2/zone_scale,y2/zone_scale,y1/zone_scale],'r-',linewidth=2,alpha=.9)
    axes[0].axis('equal')
    axes[0].axis('off')

    axes[1].imshow(np.nanmean(np.array(all_mvls),axis=0),origin='lower',cmap='viridis')
    axes[1].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
            [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=2,alpha=.9)
    axes[1].axis('equal')
    axes[1].axis('off')
    
    fig.savefig(fig2_dir + '/panel_2F.png', dpi=400)
    plt.close()
            
    ''' panel 2G '''

    #load data for example cells
    with open(data_dir + '/tetrode/panel_2G.pickle','rb') as f:
        data_2G = pickle.load(f)    
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_2G['cell_%i' % i]['tracking']
        cell_mvls = data_2G['cell_%i' % i]['mvls']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #MVL plots
        
        #scale zone coordinates to 20 x 20 grid
        zone_scale = 120./20.
        
        axes[1, i].imshow(cell_mvls['mvls'],vmin=0,vmax = np.max(cell_mvls['mvls']),cmap='viridis',origin='lower') 
        axes[1, i].scatter(cell_mvls['max_x'],cell_mvls['max_y'],marker='x',color='k',s=100,linewidth=1,clip_on=False,zorder=1000)
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig2_dir + '/panel_2G.png', dpi=400)
    plt.close()
    
    
    ''' panel 2H '''
    
    #load all MVL data
    with open(data_dir + '/tetrode/conj_baseline_mvls.pickle','rb') as f:
        baseline_mvls = pickle.load(f)
        
    #consolidate MVL maps and MVLmax locations across all pure CB cells
    all_mvls = []
    max_x = []
    max_y = []
    
    for key in baseline_mvls.keys():
        all_mvls.append(baseline_mvls[key]['resampled_mvls'])
        max_x.append(baseline_mvls[key]['resampled_max_x'])
        max_y.append(baseline_mvls[key]['resampled_max_y'])

    #make plots
    fig, axes = plt.subplots(1, 2)
    counts, xedges, yedges, im = axes[0].hist2d(max_x,max_y,bins=np.arange(0,21),cmap='viridis')
    axes[0].plot([x1/zone_scale,x2/zone_scale,x2/zone_scale,x1/zone_scale,x1/zone_scale],
            [y1/zone_scale,y1/zone_scale,y2/zone_scale,y2/zone_scale,y1/zone_scale],'r-',linewidth=2,alpha=.9)
    axes[0].axis('equal')
    axes[0].axis('off')

    axes[1].imshow(np.nanmean(np.array(all_mvls),axis=0),origin='lower',cmap='viridis')
    axes[1].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
            [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=2,alpha=.9)
    axes[1].axis('equal')
    axes[1].axis('off')
    
    fig.savefig(fig2_dir + '/panel_2H.png', dpi=400)
    plt.close()
    
    
    ''' panel 2I '''
    
    #load data for example cells
    with open(data_dir + '/tetrode/panel_2I.pickle','rb') as f:
        data_2I = pickle.load(f)
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_2I['cell_%i' % i]['tracking']
        cell_heatmap = data_2I['cell_%i' % i]['heatmap']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c='black',zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #heatmaps
        
        #scale zone coordinates to 48 x 48 heatmap
        zone_scale = 120./48.
        
        axes[1, i].imshow(np.flipud(cell_heatmap), vmin=0, vmax=np.ceil(np.nanmax(cell_heatmap)), origin='lower', cmap='viridis')
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig2_dir + '/panel_2I.png', dpi=400)
    plt.close()
    
    
    ''' panel 2J '''
    
    #load all grid cell heatmaps
    with open(data_dir + '/tetrode/grid_baseline_heatmaps.pickle','rb') as f:
        baseline_heatmaps = pickle.load(f)

    #normalize heatmaps and calculate field centroid locations
    normed_heatmaps = []
    field_centers = []
    for key in baseline_heatmaps.keys():

        #normalize heatmap
        heatmap_copy = np.flipud(copy.deepcopy(baseline_heatmaps[key]['resampled_heatmap']))
        normed_heatmap = heatmap_copy - np.nanmin(heatmap_copy)
        normed_heatmap = heatmap_copy / np.nanmax(heatmap_copy)
        normed_heatmaps.append(normed_heatmap)
        
        #bin size in sq cm (so we can count only fields with size > 150 sq cm)
        bin_size = (120. / float(48))**2
    
        #threshold to 30% of max
        threshed = np.flipud(copy.deepcopy(baseline_heatmaps[key]['resampled_heatmap']))
        threshed[threshed<.3*np.nanmax(threshed)]=0
        
        #isolate nodes > 150 sq cm in area
        nodes,num_nodes = ndimage.label(threshed)
        for k in range(1,num_nodes+1):
            if np.sum(nodes==k) * bin_size < 150:
                nodes[nodes==k] = 0
                
        #compute center of mass of each node
        full_nodes = np.unique(nodes)
        center_of_mass = []
        for k in full_nodes:
            if k != 0:
                only_k = copy.deepcopy(nodes)
                only_k[nodes!=k] = 0
                only_k[nodes==k] = 1
                center_of_mass.append(ndimage.center_of_mass(only_k))
                
        #add to list
        field_centers += list(center_of_mass)

    #scale to match bins to cm
    field_centers = 1.25 + 2.5 * np.array(field_centers).T
    zone_scale = 120./48.
    
    #make plots
    fig, axes = plt.subplots(1, 2)
    axes[0].scatter(np.array(field_centers[1]),np.array(field_centers[0]), marker='x', s=20, linewidth=1, c='black')
    axes[0].plot([x1,x2,x2,x1,x1],
            [y1,y1,y2,y2,y1],'r-',linewidth=2,alpha=.9)
    axes[0].plot([0,120,120,0,0],[0,0,120,120,0],'k-')
    axes[0].axis('equal')
    axes[0].axis('off')

    axes[1].imshow(np.nanmean(np.array(normed_heatmaps),axis=0),origin='lower',cmap='viridis')
    axes[1].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
            [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=2,alpha=.9)
    axes[1].axis('equal')
    axes[1].axis('off')
    
    fig.savefig(fig2_dir + '/panel_2J.png', dpi=400)
    plt.close()
    
    
    
    ''' Figure 3 '''

    fig3_dir = fig_dir + '/fig3'
    if not os.path.exists(fig3_dir):
        os.makedirs(fig3_dir)
        
    ''' panel 3A '''

    #load example behavior tracking
    with open(data_dir + '/tetrode/panel_3A.pickle','rb') as f:
        behavior_dict = pickle.load(f)
        
    #load behavioral occupancies
    with open(data_dir + '/tetrode/session_occupancies.pickle','rb') as f:
        occ_dict = pickle.load(f)
        
    #compute the mean of all occupancies for this session
    s1_occs = []
    s2_occs = []
    s3_occs = []
    for key in occ_dict['cue_east'].keys():
        s1_occs.append(occ_dict['cue_east'][key]['session_0'])
        s2_occs.append(occ_dict['cue_east'][key]['session_1'])
        s3_occs.append(occ_dict['cue_east'][key]['session_2'])

    s1_mean = np.nanmean(np.array(s1_occs),axis=0) / framerate
    s2_mean = np.nanmean(np.array(s2_occs),axis=0) / framerate
    s3_mean = np.nanmean(np.array(s3_occs),axis=0) / framerate
    
    overall_max = np.round(np.nanmax([np.nanmax(s1_mean),np.nanmax(s2_mean),np.nanmax(s3_mean)]))
    
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
    
    zone_colors = ['red','black','black','black']
    zone_scale = 120./30.
    
    fig, axes = plt.subplots(2, 3)
    
    for i in range(3):
        
        axes[0, i].plot(behavior_dict['session_%i' % i]['pos_x'],behavior_dict['session_%i' % i]['pos_y'], 'gray', alpha=.6, clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-')
        axes[0, i].axis('equal')
        axes[0, i].axis('off')
        

        if i == 0:
            axes[1, i].imshow(s1_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 1:
            axes[1, i].imshow(s2_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 2:
            axes[1, i].imshow(s3_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
            
        axes[1, i].plot([27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5,27.5/zone_scale - 0.5],
                [67.5/zone_scale - 0.5,67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5],color=zone_colors[0],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5,37.5/zone_scale - 0.5],
                [27.5/zone_scale - 0.5,27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5],color=zone_colors[1],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5,77.5/zone_scale - 0.5],
                [37.5/zone_scale - 0.5,37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5],color=zone_colors[2],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5,67.5/zone_scale - 0.5],
                [77.5/zone_scale - 0.5,77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5],color=zone_colors[3],linestyle='-',linewidth=1,alpha=.75)
        
        axes[1, i].axis('off')
    
    fig.savefig(fig3_dir + '/panel_3A.png', dpi=400)
    plt.close()
    
    
    ''' panel 3B '''
    
    #load data
    behavior_data = pd.read_csv(data_dir + '/tetrode/behavior_csv.csv')
    
    #find post-training days with 3 full sessions
    unique_sessions = np.unique(behavior_data['animalsession'])
    all3_data = pd.DataFrame(columns=behavior_data.columns)
    for s in unique_sessions:
        s_data = behavior_data[behavior_data['animalsession']==s]
        if len(s_data) == 3 and 'training_sessions' not in s_data['session']:
            all3_data = pd.concat([all3_data,s_data],ignore_index=True)
            
    all3_data = all3_data[all3_data['session'] != 'training_sessions']
    
    #and now just those for the CCW rotation session
    pre_data = all3_data[(all3_data['exp_type']=='cue_east')&(all3_data['session_num']==1)]
    session_data = all3_data[(all3_data['exp_type']=='cue_east')&(all3_data['session_num']==2)]
    post_data = all3_data[(all3_data['exp_type']=='cue_east')&(all3_data['session_num']==3)]
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'entries_per_minute')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'entries_per_minute')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'entries_per_minute')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3B.png', dpi=400)
    plt.close()
    
    
    ''' panel 3C '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_entry_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_entry_ind']),np.nanmean(pre_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_entry_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_entry_ind']),np.nanmean(session_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_entry_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_entry_ind']),np.nanmean(post_data['true_entry_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3C.png')
    plt.close()
    
    
    ''' panel 3D '''
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'occ')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'occ')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'occ')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3D.png', dpi=400)
    plt.close()
    
    
    ''' panel 3E '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_occ_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_occ_ind']),np.nanmean(pre_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_occ_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_occ_ind']),np.nanmean(session_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_occ_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_occ_ind']),np.nanmean(post_data['true_occ_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3E.png')
    plt.close()
        
    ''' panel 3F '''

    #load example behavior tracking
    with open(data_dir + '/tetrode/panel_3F.pickle','rb') as f:
        behavior_dict = pickle.load(f)
        
    #load behavioral occupancies
    with open(data_dir + '/tetrode/session_occupancies.pickle','rb') as f:
        occ_dict = pickle.load(f)
        
    #compute the mean of all occupancies for this session
    s1_occs = []
    s2_occs = []
    s3_occs = []
    for key in occ_dict['cue_west'].keys():
        s1_occs.append(occ_dict['cue_west'][key]['session_0'])
        s2_occs.append(occ_dict['cue_west'][key]['session_1'])
        s3_occs.append(occ_dict['cue_west'][key]['session_2'])

    s1_mean = np.nanmean(np.array(s1_occs),axis=0) / framerate
    s2_mean = np.nanmean(np.array(s2_occs),axis=0) / framerate
    s3_mean = np.nanmean(np.array(s3_occs),axis=0) / framerate
    
    overall_max = np.round(np.nanmax([np.nanmax(s1_mean),np.nanmax(s2_mean),np.nanmax(s3_mean)]))
    
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
    
    zone_colors = ['red','black','black','black']
    zone_scale = 120./30.
    
    fig, axes = plt.subplots(2, 3)
    
    for i in range(3):
        
        axes[0, i].plot(behavior_dict['session_%i' % i]['pos_x'],behavior_dict['session_%i' % i]['pos_y'], 'gray', alpha=.6, clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-')
        axes[0, i].axis('equal')
        axes[0, i].axis('off')
        

        if i == 0:
            axes[1, i].imshow(s1_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 1:
            axes[1, i].imshow(s2_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 2:
            axes[1, i].imshow(s3_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
            
        axes[1, i].plot([27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5,27.5/zone_scale - 0.5],
                [67.5/zone_scale - 0.5,67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5],color=zone_colors[0],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5,37.5/zone_scale - 0.5],
                [27.5/zone_scale - 0.5,27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5],color=zone_colors[1],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5,77.5/zone_scale - 0.5],
                [37.5/zone_scale - 0.5,37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5],color=zone_colors[2],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5,67.5/zone_scale - 0.5],
                [77.5/zone_scale - 0.5,77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5],color=zone_colors[3],linestyle='-',linewidth=1,alpha=.75)
        
        axes[1, i].axis('off')
    
    fig.savefig(fig3_dir + '/panel_3F.png', dpi=400)
    plt.close()
    
    
    ''' panel 3G '''
    
    #data from the CW rotation session
    pre_data = all3_data[(all3_data['exp_type']=='cue_west')&(all3_data['session_num']==1)]
    session_data = all3_data[(all3_data['exp_type']=='cue_west')&(all3_data['session_num']==2)]
    post_data = all3_data[(all3_data['exp_type']=='cue_west')&(all3_data['session_num']==3)]
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'entries_per_minute')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'entries_per_minute')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'entries_per_minute')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3G.png', dpi=400)
    plt.close()
    
    
    ''' panel 3H '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_entry_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_entry_ind']),np.nanmean(pre_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_entry_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_entry_ind']),np.nanmean(session_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_entry_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_entry_ind']),np.nanmean(post_data['true_entry_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3H.png')
    plt.close()
    
    
    ''' panel 3I '''
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'occ')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'occ')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'occ')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3I.png', dpi=400)
    plt.close()
    
    
    ''' panel 3J '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_occ_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_occ_ind']),np.nanmean(pre_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_occ_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_occ_ind']),np.nanmean(session_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_occ_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_occ_ind']),np.nanmean(post_data['true_occ_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3J.png')
    plt.close()
    
        
    ''' panel 3K '''

    #load example behavior tracking
    with open(data_dir + '/tetrode/panel_3K.pickle','rb') as f:
        behavior_dict = pickle.load(f)
        
    #load behavioral occupancies
    with open(data_dir + '/tetrode/session_occupancies.pickle','rb') as f:
        occ_dict = pickle.load(f)
        
    #compute the mean of all occupancies for this session
    s1_occs = []
    s2_occs = []
    s3_occs = []
    for key in occ_dict['two_cue'].keys():
        s1_occs.append(occ_dict['two_cue'][key]['session_0'])
        s2_occs.append(occ_dict['two_cue'][key]['session_1'])
        s3_occs.append(occ_dict['two_cue'][key]['session_2'])

    s1_mean = np.nanmean(np.array(s1_occs),axis=0) / framerate
    s2_mean = np.nanmean(np.array(s2_occs),axis=0) / framerate
    s3_mean = np.nanmean(np.array(s3_occs),axis=0) / framerate
    
    overall_max = np.round(np.nanmax([np.nanmax(s1_mean),np.nanmax(s2_mean),np.nanmax(s3_mean)]))
    
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
    
    zone_colors = ['red','black','black','black']
    zone_scale = 120./30.
    
    fig, axes = plt.subplots(2, 3)
    
    for i in range(3):
        
        axes[0, i].plot(behavior_dict['session_%i' % i]['pos_x'],behavior_dict['session_%i' % i]['pos_y'], 'gray', alpha=.6, clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-')
        axes[0, i].axis('equal')
        axes[0, i].axis('off')
        

        if i == 0:
            axes[1, i].imshow(s1_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 1:
            axes[1, i].imshow(s2_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
        elif i == 2:
            axes[1, i].imshow(s3_mean.T, vmin=0, vmax=overall_max, cmap='viridis', origin='lower')
            
        axes[1, i].plot([27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5,27.5/zone_scale - 0.5],
                [67.5/zone_scale - 0.5,67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5],color=zone_colors[0],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5,37.5/zone_scale - 0.5],
                [27.5/zone_scale - 0.5,27.5/zone_scale - 0.5,42.5/zone_scale - 0.5,42.5/zone_scale - 0.5,27.5/zone_scale - 0.5],color=zone_colors[1],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5,77.5/zone_scale - 0.5],
                [37.5/zone_scale - 0.5,37.5/zone_scale - 0.5,52.5/zone_scale - 0.5,52.5/zone_scale - 0.5,37.5/zone_scale - 0.5],color=zone_colors[2],linestyle='-',linewidth=1,alpha=.75)
        axes[1, i].plot([67.5/zone_scale - 0.5,82.5/zone_scale - 0.5,82.5/zone_scale - 0.5,67.5/zone_scale - 0.5,67.5/zone_scale - 0.5],
                [77.5/zone_scale - 0.5,77.5/zone_scale - 0.5,92.5/zone_scale - 0.5,92.5/zone_scale - 0.5,77.5/zone_scale - 0.5],color=zone_colors[3],linestyle='-',linewidth=1,alpha=.75)
        
        axes[1, i].axis('off')
    
    fig.savefig(fig3_dir + '/panel_3K.png', dpi=400)
    plt.close()
    
    
    ''' panel 3L '''
    
    #data from the CW rotation session
    pre_data = all3_data[(all3_data['exp_type']=='two_cue')&(all3_data['session_num']==1)]
    session_data = all3_data[(all3_data['exp_type']=='two_cue')&(all3_data['session_num']==2)]
    post_data = all3_data[(all3_data['exp_type']=='two_cue')&(all3_data['session_num']==3)]
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'entries_per_minute')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'entries_per_minute')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'entries_per_minute')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3L.png', dpi=400)
    plt.close()
    
    
    ''' panel 3M '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_entry_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_entry_ind']),np.nanmean(pre_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_entry_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_entry_ind']),np.nanmean(session_data['true_entry_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_entry_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_entry_ind']),np.nanmean(post_data['true_entry_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3M.png')
    plt.close()
    
    
    ''' panel 3N '''
    
    jitter = .10 * np.random.normal(size=len(session_data))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    session_count = 0
    
    maxes = []

    for session in ['pre','exp','post']:
        
        if session == 'pre':
            rel_data = pre_data
        elif session == 'exp':
            rel_data = session_data
        elif session == 'post':
            rel_data = post_data
        
        zones = ['true','cw','opp','ccw']

        edgecolors = ['black','gray','gray','gray']
        if session == 'pre':
            facecolors = ['red','red','red','red']
        elif session == 'exp':
            facecolors = ['blue','blue','blue','blue']
        elif session == 'post':
            facecolors = ['green','green','green','green']
                
        count = 0
        for i in range(len(zones)):
            zone=zones[i]
            color=colors[i]
            mean_val = np.mean(rel_data['%s_%s' % (zone,'occ')])
            maxes.append(np.max(rel_data['%s_%s' % (zone,'occ')]))
            ax.plot([session_count + 3.5*count-.37, session_count + 3.5*count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            ax.scatter(session_count + 3.5*count+jitter,rel_data['%s_%s' % (zone,'occ')],c=facecolors[i],edgecolors=edgecolors[i],clip_on=False)
            count += 1
            
        session_count += 1
            
    ax.plot([-.5,13],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,13])
    ax.set_ylim([0,1.2*np.max(maxes)])
    ax.set_xticks([1,4.5,8,11.5],['NW','SW','SE','NE'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.9*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3N.png', dpi=400)
    plt.close()
    
    
    ''' panel 3O '''
    
    #plot entry indices
    fig = plt.figure()
    ax = fig.add_subplot(111)
    jitter = .10 * np.random.normal(size=len(session_data))
    ax.scatter(jitter, pre_data['true_occ_ind'],c='red',edgecolor='black',clip_on=False)
    ax.plot([-.37,.37],[np.nanmean(pre_data['true_occ_ind']),np.nanmean(pre_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(1+jitter, session_data['true_occ_ind'],c='blue',edgecolor='black',clip_on=False)
    ax.plot([0.63,1.37],[np.nanmean(session_data['true_occ_ind']),np.nanmean(session_data['true_occ_ind'])],'k-',linewidth=3)
    ax.scatter(2+jitter, post_data['true_occ_ind'],c='green',edgecolor='black',clip_on=False)
    ax.plot([1.63,2.37],[np.nanmean(post_data['true_occ_ind']),np.nanmean(post_data['true_occ_ind'])],'k-',linewidth=3)
    ax.plot([-.5, 2.5],[0,0],'k-')
    ax.plot([-.5, 2.5], [.25, .25], 'r--')
        
    ax.set_xticks([0,1,2])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,.9])
    ax.set_yticks([0,.25, .5, .75])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.4*(x1-x0)/(y1-y0))
    fig.savefig(fig3_dir + '/panel_3O.png', dpi=400)
    plt.close()
    
    
    ''' Figure 4 '''
    
    fig4_dir = fig_dir + '/fig4'
    if not os.path.exists(fig4_dir):
        os.makedirs(fig4_dir)
        
    ''' panel 4A '''

    #load tuning curves
    with open(data_dir + '/tetrode/panel_4A.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','Cue east','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(15. - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([15,15],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig4_dir + '/panel_4A.png', dpi=400)
    plt.close()
    
    
    ''' panel 4B '''

    #load tuning curves
    with open(data_dir + '/tetrode/panel_4B.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','Cue west','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(15. - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([15,15],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig4_dir + '/panel_4B.png', dpi=400)
    plt.close()
    
    
    ''' panel 4C '''
    
    #load data for example cell
    with open(data_dir + '/tetrode/panel_4C.pickle','rb') as f:
        data_4C = pickle.load(f)    
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cell
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2
    
    maxes = []
    for i in range(3):
        maxes.append(np.nanmax(data_4C['session_%i' % i]['mvls']['mvls']))
        
    overmax = np.nanmax(maxes)

    for i in range(3):
        cell_tracking = data_4C['session_%i' % i]['tracking']
        cell_mvls = data_4C['session_%i' % i]['mvls']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #MVL plots
        
        #scale zone coordinates to 20 x 20 grid
        zone_scale = 120./20.
        
        axes[1, i].imshow(cell_mvls['mvls'],vmin=0,vmax = overmax,cmap='viridis',origin='lower') 
        axes[1, i].scatter(cell_mvls['max_x'],cell_mvls['max_y'],marker='x',color='k',s=100,linewidth=1,clip_on=False,zorder=1000)
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig4_dir + '/panel_4C.png', dpi=400)
    plt.close()
    
    

    ''' panel 4D '''
    
    #load data for example cell
    with open(data_dir + '/tetrode/panel_4D.pickle','rb') as f:
        data_4D = pickle.load(f)    
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cell
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    maxes = []
    for i in range(3):
        maxes.append(np.nanmax(data_4D['session_%i' % i]['mvls']['mvls']))
        
    overmax = np.nanmax(maxes)

    for i in range(3):
        cell_tracking = data_4D['session_%i' % i]['tracking']
        cell_mvls = data_4D['session_%i' % i]['mvls']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #MVL plots
        
        #scale zone coordinates to 20 x 20 grid
        zone_scale = 120./20.
        
        axes[1, i].imshow(cell_mvls['mvls'],vmin=0,vmax = overmax,cmap='viridis',origin='lower') 
        axes[1, i].scatter(cell_mvls['max_x'],cell_mvls['max_y'],marker='x',color='k',s=100,linewidth=1,clip_on=False,zorder=1000)
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig4_dir + '/panel_4D.png', dpi=400)
    plt.close()
    
    
    ''' panel 4E '''
    
    #load HD shifts
    with open(data_dir + '/tetrode/panel_4E.pickle','rb') as f:
        data_4E = pickle.load(f)    
    
    west_shifts = data_4E['cue_west']
    east_shifts = data_4E['cue_east']
    polar_dot_plot(np.concatenate([west_shifts,east_shifts]),labels=np.array(['west']*len(west_shifts) + ['east']*len(east_shifts)),savefile=fig4_dir + '/panel_4E.png')


    ''' panel 4F '''
    
    #load HD shifts
    with open(data_dir + '/tetrode/panel_4F.pickle','rb') as f:
        data_4F = pickle.load(f)    
    
    west_shifts = data_4F['cue_west']
    east_shifts = data_4F['cue_east']
    polar_dot_plot(np.concatenate([west_shifts,east_shifts]),labels=np.array(['west']*len(west_shifts) + ['east']*len(east_shifts)),savefile=fig4_dir + '/panel_4F.png')


    ''' panel 4G '''
    
    #load tuning curves
    with open(data_dir + '/tetrode/panel_4G.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','South/north','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(8 - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([7.5,7.5],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig4_dir + '/panel_4G.png', dpi=400)
    plt.close()
    
    
    ''' panel 4H '''
    
    #load bidirectionality indices
    with open(data_dir + '/tetrode/panel_4H.pickle','rb') as f:
        data_4H = pickle.load(f)

    exp = np.array(data_4H['exp']) - np.array(data_4H['pre'])
    ctl = np.array(data_4H['post']) - np.array(data_4H['pre'])

    data_dict = {}
    data_dict['val'] = np.concatenate((exp,ctl))
    data_dict['label'] = ['S/N - S1'] * len(exp) + ['S2 - S1']*len(ctl)
    data_df = pd.DataFrame(data_dict)
    
    plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['red','gray'],edgecolor='black',linewidth=1,clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_xlim([-.5,len(np.unique(data_df['label']))-.5])

    ax.set_ylim([-1,1.5])
    ax.set_yticks([-1,-.5,0,.5,1,1.5])
        
    ax.set_ylabel('delta BI')
    ax.set_xlabel('')
    
    median_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()
        median_val = data_df[data_df['label']==var].val.mean()
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=10,alpha=0.8)
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.savefig(fig4_dir + '/panel_4H.png', dpi=400)
    plt.close()
    
    
    ''' panel 4I '''
    
    #load modulation indices
    with open(data_dir + '/tetrode/ab_mods.pickle','rb') as f:
        mods = pickle.load(f)

    og_mods = mods['POR']['og']
    new_mods = mods['POR']['new']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(og_mods,new_mods,zorder=1000,c='k',clip_on=False)
    ax.plot((0,1),(0,1),'k-')
    ax.set_ylabel('Cue north mod index')
    ax.set_xlabel('Cue south mod index')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(fig4_dir + '/panel_4I.png', dpi=400)
    plt.close()
    
    
    ''' panel 4J '''
    
    #load data for example cell
    with open(data_dir + '/tetrode/panel_4J.pickle','rb') as f:
        data_4J = pickle.load(f)    
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cell
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    maxes = []
    for i in range(3):
        maxes.append(np.nanmax(data_4J['session_%i' % i]['mvls']['mvls']))
        
    overmax = np.nanmax(maxes)

    for i in range(3):
        cell_tracking = data_4J['session_%i' % i]['tracking']
        cell_mvls = data_4J['session_%i' % i]['mvls']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'k-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #MVL plots
        
        #scale zone coordinates to 20 x 20 grid
        zone_scale = 120./20.
        
        axes[1, i].imshow(cell_mvls['mvls'],vmin=0,vmax = overmax,cmap='viridis',origin='lower') 
        axes[1, i].scatter(cell_mvls['max_x'],cell_mvls['max_y'],marker='x',color='k',s=100,linewidth=1,clip_on=False,zorder=1000)
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig4_dir + '/panel_4J.png', dpi=400)
    plt.close()
    
    
    ''' panel 4K '''
    
    #load MVL distances
    with open(data_dir + '/tetrode/panel_4K.pickle','rb') as f:
        data_4K = pickle.load(f)    
    
    exp = (np.array(data_4K['exp']) - np.array(data_4K['pre'])) * 6.
    std = (np.array(data_4K['post']) - np.array(data_4K['pre'])) * 6.
    
    all_dists = [exp,std]
    colors=['red','gray']
    edgecolors = ['black','black']
    jitter = .05 * np.random.normal(size=len(exp))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    count = 0
    for i in range(len(all_dists)):
        color=colors[i]
        mean_val = np.mean(all_dists[i])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,all_dists[i],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,1.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,1.5])
    ax.set_ylim([-80,50])
    ax.set_yticks([-50,0,50])
    ax.set_xticks([0,1],['S/N - S1','S2 - S1'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    fig.savefig(fig4_dir + '/panel_4K.png', dpi=400)
    plt.close()
    
    
    
    ''' Figure 5 '''
    
    fig5_dir = fig_dir + '/fig5'
    if not os.path.exists(fig5_dir):
        os.makedirs(fig5_dir)
        
    ''' panel 5A '''

    #load tuning curves
    with open(data_dir + '/tetrode/panel_5A.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','Cue east','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(15. - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([15,15],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig5_dir + '/panel_5A.png', dpi=400)
    plt.close()
    
    
    ''' panel 5B '''

    #load tuning curves
    with open(data_dir + '/tetrode/panel_5B.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','Cue west','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(15. - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([15,15],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig5_dir + '/panel_5B.png', dpi=400)
    plt.close()
    
    
    ''' panel 5C '''
    
    #load data for example cells
    with open(data_dir + '/tetrode/panel_5C.pickle','rb') as f:
        data_5C = pickle.load(f)
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_5C['session_%i' % i]['tracking']
        cell_heatmap = data_5C['session_%i' % i]['heatmap']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c='black',zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #heatmaps
        
        #scale zone coordinates to 48 x 48 heatmap
        zone_scale = 120./48.
        
        axes[1, i].imshow(np.flipud(cell_heatmap), vmin=0, vmax=np.ceil(np.nanmax(cell_heatmap)), origin='lower', cmap='viridis')
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig5_dir + '/panel_5C.png', dpi=400)
    plt.close()
    
    
    ''' panel 5D '''
    
    #load data for example cells
    with open(data_dir + '/tetrode/panel_5D.pickle','rb') as f:
        data_5D = pickle.load(f)
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_5D['session_%i' % i]['tracking']
        cell_heatmap = data_5D['session_%i' % i]['heatmap']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c='black',zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #heatmaps
        
        #scale zone coordinates to 48 x 48 heatmap
        zone_scale = 120./48.
        
        axes[1, i].imshow(np.flipud(cell_heatmap), vmin=0, vmax=np.ceil(np.nanmax(cell_heatmap)), origin='lower', cmap='viridis')
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig5_dir + '/panel_5D.png', dpi=400)
    plt.close()
    
    
    ''' panel 5E '''
    
    #load HD shifts
    with open(data_dir + '/tetrode/panel_5E.pickle','rb') as f:
        data_5E = pickle.load(f)    
    
    west_shifts = data_5E['cue_west']
    east_shifts = data_5E['cue_east']
    polar_dot_plot(np.concatenate([west_shifts,east_shifts]),labels=np.array(['west']*len(west_shifts) + ['east']*len(east_shifts)),savefile=fig5_dir + '/panel_5E.png')


    ''' panel 5F '''
    
    #load ratemap correlations
    with open(data_dir + '/tetrode/panel_5FK.pickle','rb') as f:
        data_5FK = pickle.load(f)    
    
    fig, axes = plt.subplots(1,2)
    
    sessions = ['cue_east','cue_west']
    
    for s in range(2):
        
        session = sessions[s]
        rel_data = data_5FK[session]

        rotations = ['standard','ccw','opp','cw']
        colors = ['red','gray','gray','gray']
        edgecolors = ['black','black','black','black']

        jitter = .1 * np.random.normal(size=len(rel_data['standard']))

        count = 0
        for i in range(len(rotations)):
            rotation=rotations[i]
            color=colors[i]
            mean_val = np.mean(rel_data[rotation])
            axes[s].plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
            axes[s].scatter(count+jitter,rel_data[rotation],c=color,edgecolors=edgecolors[i],clip_on=False)
            count += 1
        axes[s].plot([-.5,3.5],[0,0],'k-',zorder=-1)
        axes[s].set_xlim([-.5,3.5])
        axes[s].set_ylim([-.55,1])
        axes[s].set_xticks([0,1,2,3],['0','90','180','270'])
        axes[s].set_yticks([-0.5,0,0.5,1])
        axes[s].spines['right'].set_visible(False)
        axes[s].spines['top'].set_visible(False)
        axes[s].spines['left'].set_visible(True)
        axes[s].spines['bottom'].set_visible(True)
        axes[s].set_ylabel('Correlation with S1 rate map')
        axes[s].set_xlabel('Rotation (deg)')
        x0,x1 = axes[s].get_xlim()
        y0,y1 = axes[s].get_ylim()
        axes[s].set_aspect(1.2*(x1-x0)/(y1-y0))
        plt.tight_layout()
        
    fig.savefig(fig5_dir + '/panel_5F.png', dpi=400)
    plt.close()
        
        

    ''' panel 5G '''
    
    #load tuning curves
    with open(data_dir + '/tetrode/panel_5G.pickle','rb') as f:
        curve_dict = pickle.load(f)
        
    fig, axes = plt.subplots(1,3)
    
    sessions = ['pre','exp','post']
    titles = ['Cue south','South/north','Cue south']
    for s in range(3):
        session = sessions[s]
        rates = curve_dict[session]
        pfds = curve_dict['pfds']
        recentered_rates = copy.deepcopy(rates)
            
        bin_centers = np.array(range(6,366,12))
        for i in range(len(rates)):
            abs_diffs = abs((bin_centers - pfds[i] + 180) % 360 - 180)
            pfd_bin = np.argmin(abs_diffs)
            recentered_rates[i] = np.roll(rates[i],int(8 - pfd_bin))
    
        recentered_rates = np.concatenate((recentered_rates,recentered_rates[:,0].reshape((len(recentered_rates),1))),axis=1)

        axes[s].imshow(recentered_rates,cmap='viridis',aspect=45./len(rates))
        axes[s].set_title(titles[s])
        axes[s].set_xticks([0,7.5,15,22.5,30])
        axes[s].plot([7.5,7.5],[-0.5,len(recentered_rates)-.5],'r-',linewidth=3,zorder=1000)
        axes[s].set_xticklabels(['','-90','0','90',''])
        axes[s].set_yticks([0,len(rates)-1])
        axes[s].set_xlabel('Rel. HD')
        if s == 0:
            axes[s].set_yticklabels(['1',str(len(rates))])
            axes[s].set_ylabel('Cell number')
        else:
            axes[s].set_yticklabels([])
            
    fig.savefig(fig5_dir + '/panel_45.png', dpi=400)
    plt.close()
    
    
    ''' panel 5H '''
    
    #load bidirectionality indices
    with open(data_dir + '/tetrode/panel_5H.pickle','rb') as f:
        data_5H = pickle.load(f)

    exp = np.array(data_5H['exp']) - np.array(data_5H['pre'])
    ctl = np.array(data_5H['post']) - np.array(data_5H['pre'])

    data_dict = {}
    data_dict['val'] = np.concatenate((exp,ctl))
    data_dict['label'] = ['S/N - S1'] * len(exp) + ['S2 - S1']*len(ctl)
    data_df = pd.DataFrame(data_dict)
    
    plt.figure()
    ax = sns.stripplot(x='label',y='val',data=data_df,jitter=True,palette=['red','gray'],edgecolor='black',linewidth=1,clip_on=False,size=6,alpha=0.8)
    ax.plot((-1,len(np.unique(data_df['label']))),(0,0),'k-',alpha=.8)
    ax.set_xlim([-.5,len(np.unique(data_df['label']))-.5])

    ax.set_ylim([-1,1.5])
    ax.set_yticks([-1,-.5,0,.5,1,1.5])
        
    ax.set_ylabel('delta BI')
    ax.set_xlabel('')
    
    median_width = 0.5

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        var = text.get_text()
        median_val = data_df[data_df['label']==var].val.mean()
        ax.plot([tick-median_width/2, tick+median_width/2], [median_val, median_val],
                lw=4, color='k',zorder=10,alpha=0.8)
    
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1.3*(x1-x0)/(y1-y0))
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    plt.savefig(fig5_dir + '/panel_5H.png', dpi=400)
    plt.close()
    
    
    ''' panel 5I '''
    
    #load modulation indices
    with open(data_dir + '/tetrode/ab_mods.pickle','rb') as f:
        mods = pickle.load(f)

    og_mods = mods['MEC']['og']
    new_mods = mods['MEC']['new']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(og_mods,new_mods,zorder=1000,c='k',clip_on=False)
    ax.plot((0,1),(0,1),'k-')
    ax.set_ylabel('Cue north mod index')
    ax.set_xlabel('Cue south mod index')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.5,1])
    ax.set_yticks([0,0.5,1])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    fig.savefig(fig5_dir + '/panel_5I.png', dpi=400)
    plt.close()
    
    
    ''' panel 5J '''
    
    #load data for example cell
    with open(data_dir + '/tetrode/panel_5J.pickle','rb') as f:
        data_5J = pickle.load(f)
        
    #approximate zone edge coordinates for plotting
    x1 = 27.5
    x2 = 42.5
    y1 = 67.5
    y2 = 82.5
        
    #make plots for example cells
    fig, axes = plt.subplots(2, 3)
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2

    for i in range(3):
        cell_tracking = data_5J['session_%i' % i]['tracking']
        cell_heatmap = data_5J['session_%i' % i]['heatmap']

        #path/spike plots
        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        
        axes[0, i].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[0, i].scatter(spike_x,spike_y,s=markersize,c='black',zorder=1,clip_on=False)
        axes[0, i].plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],'r-',linewidth=1,zorder=1000,alpha=0.9)
        axes[0, i].axis('off')
        axes[0, i].axis('equal')
        
        #heatmaps
        
        #scale zone coordinates to 48 x 48 heatmap
        zone_scale = 120./48.
        
        axes[1, i].imshow(np.flipud(cell_heatmap), vmin=0, vmax=np.ceil(np.nanmax(cell_heatmap)), origin='lower', cmap='viridis')
        axes[1, i].plot([x1/zone_scale - 0.5,x2/zone_scale - 0.5,x2/zone_scale - 0.5,x1/zone_scale - 0.5,x1/zone_scale - 0.5],
                [y1/zone_scale - 0.5,y1/zone_scale - 0.5,y2/zone_scale - 0.5,y2/zone_scale - 0.5,y1/zone_scale - 0.5],'r-',linewidth=1,zorder=100,alpha=.9)
        axes[1, i].axis('off')
        axes[1, i].axis('equal')
    
    fig.savefig(fig5_dir + '/panel_5J.png', dpi=400)
    plt.close()
    
    
    ''' panel 5K '''
    
    #load ratemap correlations
    with open(data_dir + '/tetrode/panel_5FK.pickle','rb') as f:
        data_5FK = pickle.load(f)    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    rel_data = data_5FK['two_cue']

    rotations = ['standard','ccw','opp','cw']
    colors = ['red','gray','gray','gray']
    edgecolors = ['black','black','black','black']

    jitter = .1 * np.random.normal(size=len(rel_data['standard']))

    count = 0
    for i in range(len(rotations)):
        rotation=rotations[i]
        color=colors[i]
        mean_val = np.mean(rel_data[rotation])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,rel_data[rotation],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,3.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([-.55,1])
    ax.set_xticks([0,1,2,3],['0','90','180','270'])
    ax.set_yticks([-0.5,0,0.5,1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel('Correlation with S1 rate map')
    ax.set_xlabel('Rotation (deg)')
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
        
    fig.savefig(fig5_dir + '/panel_5K.png', dpi=400)
    plt.close()
        
        
    ''' Figure 6 '''
    
    fig6_dir = fig_dir + '/fig6'
    if not os.path.exists(fig6_dir):
        os.makedirs(fig6_dir)
        
    ''' panel 6A '''

    #load HD rotation data
    with open(data_dir + '/tetrode/panel_6AB.pickle','rb') as f:
        rot_data = pickle.load(f)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rot_data['dist_from_mec'],rot_data['hd_diff'],c=rot_data['area_colors'],edgecolor='black',clip_on=False,zorder=100)
    ax.plot([0,0],[0,180],'r-',zorder=-1)
    ax.set_xlim([-2,1.2])
    ax.set_ylim([0,180])
    ax.set_yticks([0,90,180])
    ax.set_xticks([-2,-1,0,1],[2,1,0,-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Est dist from ventral POR border (mm)')
    ax.set_ylabel('Absolute preferred HD shift (deg)')
    fig.savefig(fig6_dir + '/panel_6A.png', dpi=400)
    plt.close()
        
    ''' panel 6B '''
    
    model = smf.mixedlm('hd_diff ~ exposure', rot_data, groups=rot_data['brain_area'], re_formula='~exposure')
    fit = model.fit()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(rot_data['exposure'],rot_data['hd_diff'],c=rot_data['area_colors'],edgecolor='black',clip_on=False)
    ax.plot(np.arange(np.min(rot_data['exposure']),np.max(rot_data['exposure'])+1),fit.fe_params[0] + np.arange(np.min(rot_data['exposure']),np.max(rot_data['exposure'])+1) * fit.fe_params[1],'r--')
    ax.set_xlim([0,65])
    ax.set_ylim([0,180])
    ax.set_yticks([0,90,180])
    ax.set_ylabel('Absolute preferred HD shift (deg)')
    ax.set_xlabel('Task day')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    fig.savefig(fig6_dir + '/panel_6B.png', dpi=400)
    plt.close()
    
    

    ''' panel 6C '''

    #load HD duplication data
    with open(data_dir + '/tetrode/panel_6CD.pickle','rb') as f:
        dup_data = pickle.load(f)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dup_data['dist_from_mec'],dup_data['bi_diff'],c=dup_data['area_colors'],edgecolor='black',clip_on=False,zorder=100)
    ax.plot([0,0],[-.55,1.5],'r-',zorder=-1)
    ax.plot([-2,1.8],[0,0],'k-',zorder=-2)
    ax.set_xlim([-2,1.8])
    ax.set_ylim([-.55,1.5])
    ax.set_yticks([-.5,0,.5,1,1.5])
    ax.set_xticks([-2,-1,0,1],[2,1,0,-1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.set_xlabel('Est dist from ventral POR border (mm)')
    ax.set_ylabel('Delta bidirectionality index')
    fig.savefig(fig6_dir + '/panel_6C.png', dpi=400)
    plt.close()
        
    ''' panel 6D '''
    
    model = smf.mixedlm('bi_diff ~ exposure', dup_data, groups=dup_data['brain_area'], re_formula='~exposure')
    fit = model.fit()
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dup_data['exposure'],dup_data['bi_diff'],c=dup_data['area_colors'],edgecolor='black',clip_on=False)
    ax.plot(np.arange(np.min(dup_data['exposure']),np.max(dup_data['exposure'])+1),fit.fe_params[0] + np.arange(np.min(dup_data['exposure']),np.max(dup_data['exposure'])+1) * fit.fe_params[1],'r--')
    ax.plot([0,65],[0,0],'k-',zorder=-2)
    ax.set_xlim([0,65])
    ax.set_ylim([-.55,1.5])
    ax.set_ylabel('Delta bidirectionality index')
    ax.set_xlabel('Task day')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    fig.savefig(fig6_dir + '/panel_6D.png', dpi=400)
    plt.close()
    
    
    
    ''' Figure 7 '''
    
    fig7_dir = fig_dir + '/fig7'
    if not os.path.exists(fig7_dir):
        os.makedirs(fig7_dir)
        
        
    ''' panel 7A '''
    
    #load example trajectories
    with open(data_dir + '/npx/panel_7A.pickle','rb') as f:
        data_7A = pickle.load(f)
    
    cue_locs = ['cue_north','cue_east','cue_west','cue_south']
    
    zone_coords = {'cue_south':[27.5,42.5,67.5,82.5],
                 'cue_west':[67.5,82.5,77.5,92.5],
                 'cue_north':[77.5,92.5,37.5,52.5],
                 'cue_east':[37.5,52.5,27.5,42.5]}

    fig, axes = plt.subplots(2,2,figsize=(5,5))
    
    for i in range(4):
        
        tracking_data = data_7A['session_%i' % i]
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]
        axes[int(i/2),i%2].plot(tracking_data['pos_x'],tracking_data['pos_y'],'gray',alpha=.7)
        axes[int(i/2),i%2].plot([zone[0],zone[1],zone[1],zone[0],zone[0]],[zone[2],zone[2],zone[3],zone[3],zone[2]],'k-')
        axes[int(i/2),i%2].axis('square')
        axes[int(i/2),i%2].axis('off')

    plt.tight_layout()
    fig.savefig(fig7_dir + '/panel_7A.png', dpi=400)
    plt.close()
    
    
    ''' panel 7B '''
    
    #load behavior data
    baseline_behavior = pd.read_csv(data_dir + '/npx/baseline_behavior_csv.csv')

    animal_ids = np.array(baseline_behavior['animal'])
    colors = np.full_like(animal_ids,'gray',dtype='<U16')
    colors[animal_ids=='PL123'] = 'saddlebrown'
    colors[animal_ids=='PL124'] = 'indigo'
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(baseline_behavior['exposure'],baseline_behavior['rewards']/baseline_behavior['session_length_minutes'],c=colors,s=60,clip_on=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel('Rewards per minute')
    ax.set_xlabel('Task day')
    ax.set_ylim([0,None])
    ax.set_yticks([0,.5,1,1.5,2],[0,'',1,'',2])
    ax.set_xticks([0,10,20])
    ax.set_xlim([0,25])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.8*(x1-x0)/(y1-y0))
    fig.savefig(fig7_dir + '/panel_7B.png', dpi=400)
    plt.close()


    ''' panel 7C '''

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(baseline_behavior['exposure'],baseline_behavior['rewards']/baseline_behavior['path_length_meters'],c=colors,s=60,clip_on=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.set_ylabel('Rewards per meter')
    ax.set_xlabel('Task day')
    ax.set_ylim([0,None])
    ax.set_yticks([0,0.05, 0.1])
    ax.set_xticks([0,10,20])
    ax.set_xlim([0,25])
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(.8*(x1-x0)/(y1-y0))
    fig.savefig(fig7_dir + '/panel_7C.png', dpi=400)
    plt.close()
    
    
    ''' panel 7D '''

    post_training = baseline_behavior[baseline_behavior['session']!='training_sessions']

    zones = ['true','cw','opp','ccw']
    colors = ['red','gray','gray','gray']
    edgecolors = ['black','black','black','black']
    
    jitter = .1 * np.random.normal(size=len(post_training))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    count = 0
    for i in range(len(zones)):
        zone=zones[i]
        color=colors[i]
        mean_val = np.mean(post_training['%s_%s' % (zone,'entries_per_minute')])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,post_training['%s_%s' % (zone,'entries_per_minute')],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,3.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([0,8])
    ax.set_xticks([0,1,2,3],['Goal','CCW','Opp','CW'])
    ax.set_yticks([0,2,4,6,8])
    ax.set_ylabel('Entries per minute')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    fig.savefig(fig7_dir + '/panel_7D.png', dpi=400)
    plt.close()
    
    
    ''' panel 7E '''

    jitter = .1 * np.random.normal(size=len(post_training))
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    count = 0
    for i in range(len(zones)):
        zone=zones[i]
        color=colors[i]
        mean_val = np.mean(post_training['%s_%s' % (zone,'occ')])
        ax.plot([count-.37,count+.37],[mean_val,mean_val],'k-',linewidth=3,zorder=100)
        ax.scatter(count+jitter,post_training['%s_%s' % (zone,'occ')],c=color,edgecolors=edgecolors[i],clip_on=False)
        count += 1
    ax.plot([-.5,3.5],[0,0],'k-',zorder=-1)
    ax.set_xlim([-.5,3.5])
    ax.set_ylim([0,80])
    ax.set_xticks([0,1,2,3],['Goal','CCW','Opp','CW'])
    ax.set_yticks([0,20,40,60,80])
    ax.set_ylabel('Occupancy (s)')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    x0,x1 = ax.get_xlim()
    y0,y1 = ax.get_ylim()
    ax.set_aspect(1*(x1-x0)/(y1-y0))
    fig.savefig(fig7_dir + '/panel_7E.png', dpi=400)
    plt.close()


    ''' panel 7F '''
    
    #load example cells
    with open(data_dir + '/npx/panel_7F.pickle','rb') as f:
        data_7F = pickle.load(f)
    
    cue_locs = ['cue_north','cue_east','cue_west','cue_south']
    
    zone_coords = {'cue_south':[27.5,42.5,67.5,82.5],
                 'cue_west':[67.5,82.5,77.5,92.5],
                 'cue_north':[77.5,92.5,37.5,52.5],
                 'cue_east':[37.5,52.5,27.5,42.5]}

    fig, axes = plt.subplots(2,2,figsize=(5,5))
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2
    
    for i in range(4):
        
        cell_tracking = data_7F['cell_%i' % i]
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        spike_angles = cell_tracking['angles'][cell_tracking['spike_train']>0]
        colormap = plt.get_cmap('hsv')
        norm = mplcolors.Normalize(vmin=0, vmax=360)
        axes[int(i/2),i%2].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[int(i/2),i%2].scatter(spike_x,spike_y,s=markersize,c=spike_angles,cmap=colormap,norm=norm,zorder=1,clip_on=False)
        axes[int(i/2),i%2].plot([zone[0],zone[1],zone[1],zone[0],zone[0]],[zone[2],zone[2],zone[3],zone[3],zone[2]],'k-')
        axes[int(i/2),i%2].axis('off')
        axes[int(i/2),i%2].axis('equal')

    plt.tight_layout()
    fig.savefig(fig7_dir + '/panel_7F.png', dpi=400)
    plt.close()
    
    
    ''' panel 7G '''
    
    #load ego bearing data
    with open(data_dir + '/npx/panel_7G.pickle','rb') as f:
        data_7G = pickle.load(f)
        
    cue_locs = ['cue_north','cue_east','cue_west','cue_south']
    
    zone_coords = {'cue_south':[27.5,42.5,67.5,82.5],
                 'cue_west':[67.5,82.5,77.5,92.5],
                 'cue_north':[77.5,92.5,37.5,52.5],
                 'cue_east':[37.5,52.5,27.5,42.5]}

    zone_scale = 120./20.
    
    hist_maxes = []
    
    for cue_loc in ['cue_north','cue_south','cue_east','cue_west']:
        
        mean_mvl_hist = np.nanmean(data_7G[cue_loc],axis=0)
        hist_maxes.append(np.nanmax(mean_mvl_hist))

    hist_max = np.nanmax(hist_maxes)

    fig, axes = plt.subplots(2,2,figsize=(5,5))
    
    for i in range(4):
        
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        mean_mvl_hist = np.nanmean(data_7G[cue_loc],axis=0)

        axes[int(i/2),i%2].imshow(mean_mvl_hist.T,vmin=0,vmax=hist_max,origin='lower',cmap='viridis')
        axes[int(i/2),i%2].plot([zone[0]/zone_scale - .5,zone[1]/zone_scale - .5,zone[1]/zone_scale - .5,zone[0]/zone_scale - .5,zone[0]/zone_scale - .5],
                [zone[2]/zone_scale - .5,zone[2]/zone_scale - .5,zone[3]/zone_scale - .5,zone[3]/zone_scale - .5,zone[2]/zone_scale - .5],'r-',linewidth=3,alpha=.9)
  
        axes[int(i/2),i%2].axis('equal')
        axes[int(i/2),i%2].axis('off')
    
    fig.savefig(fig7_dir + '/panel_7G.png', dpi=400)
    plt.close()
            

    ''' panel 7H '''
    
    #load ego bearing data
    with open(data_dir + '/npx/panel_7H.pickle','rb') as f:
        data_7H = pickle.load(f)

    mvl_mins = []
    mvl_maxes = []
    
    for cue_loc in ['cue_north','cue_south','cue_east','cue_west']:
        mean_mvls = np.nanmean(data_7H[cue_loc],axis=0)
        mvl_mins.append(np.nanmin(mean_mvls))
        mvl_maxes.append(np.nanmax(mean_mvls))

    mvl_min = np.nanmin(mvl_mins)
    mvl_max = np.nanmax(mvl_maxes)

    fig, axes = plt.subplots(2,2,figsize=(5,5))
    
    for i in range(4):
        
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        mean_mvls = np.nanmean(data_7H[cue_loc],axis=0)

        axes[int(i/2),i%2].imshow(mean_mvls,vmin=mvl_min,vmax=mvl_max,origin='lower',cmap='viridis')
        axes[int(i/2),i%2].plot([zone[0]/zone_scale - .5,zone[1]/zone_scale - .5,zone[1]/zone_scale - .5,zone[0]/zone_scale - .5,zone[0]/zone_scale - .5],
                [zone[2]/zone_scale - .5,zone[2]/zone_scale - .5,zone[3]/zone_scale - .5,zone[3]/zone_scale - .5,zone[2]/zone_scale - .5],'r-',linewidth=3,alpha=.9)
        axes[int(i/2),i%2].axis('equal')
        axes[int(i/2),i%2].axis('off')
    
    fig.savefig(fig7_dir + '/panel_7H.png', dpi=400)
    plt.close()

    
    ''' panel 7I '''
    
    #load example cells
    with open(data_dir + '/npx/panel_7I.pickle','rb') as f:
        data_7I = pickle.load(f)
    
    cue_locs = ['cue_north','cue_east','cue_west','cue_south']
    
    zone_coords = {'cue_south':[27.5,42.5,67.5,82.5],
                 'cue_west':[67.5,82.5,77.5,92.5],
                 'cue_north':[77.5,92.5,37.5,52.5],
                 'cue_east':[37.5,52.5,27.5,42.5]}

    fig, axes = plt.subplots(2,2,figsize=(5,5))
    markersize = (.35 * plt.rcParams['lines.markersize']) ** 2
    
    for i in range(4):
        
        cell_tracking = data_7I['cell_%i' % i]
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        spike_x = cell_tracking['pos_x'][cell_tracking['spike_train']>0]
        spike_y = cell_tracking['pos_y'][cell_tracking['spike_train']>0]
        axes[int(i/2),i%2].plot(cell_tracking['pos_x'],cell_tracking['pos_y'],color='gray',alpha=0.6,zorder=0)
        axes[int(i/2),i%2].scatter(spike_x,spike_y,s=markersize,c='black',zorder=1,clip_on=False)
        axes[int(i/2),i%2].plot([zone[0],zone[1],zone[1],zone[0],zone[0]],[zone[2],zone[2],zone[3],zone[3],zone[2]],'r-')
        axes[int(i/2),i%2].axis('off')
        axes[int(i/2),i%2].axis('equal')

    plt.tight_layout()
    fig.savefig(fig7_dir + '/panel_7I.png', dpi=400)
    plt.close()
    
    
    ''' panel 7J '''
    
    #load heatmap and field centroid data
    with open(data_dir + '/npx/panel_7JK.pickle','rb') as f:
        data_7JK = pickle.load(f)
        
    all_field_centers = data_7JK['resampled_field_centers']
        
    cue_locs = ['cue_north','cue_east','cue_west','cue_south']
    
    zone_coords = {'cue_south':[27.5,42.5,67.5,82.5],
                 'cue_west':[67.5,82.5,77.5,92.5],
                 'cue_north':[77.5,92.5,37.5,52.5],
                 'cue_east':[37.5,52.5,27.5,42.5]}
    
    zone_scale = 120./48.

    fig, axes = plt.subplots(2,2,figsize=(5,5))    
    
    for i in range(4):
        
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        for j in range(len(all_field_centers[cue_loc])):
                axes[int(i/2),i%2].scatter(all_field_centers[cue_loc][j][1]+.5,all_field_centers[cue_loc][j][0]+.5,marker='x',c='k',s=6,clip_on=False)
        axes[int(i/2),i%2].plot([zone[0]/zone_scale,zone[1]/zone_scale,zone[1]/zone_scale,zone[0]/zone_scale,zone[0]/zone_scale],
                [zone[2]/zone_scale,zone[2]/zone_scale,zone[3]/zone_scale,zone[3]/zone_scale,zone[2]/zone_scale],'r-',linewidth=3,alpha=.9)
        axes[int(i/2),i%2].plot([0,48,48,0,0],[0,0,48,48,0],'k-')
        axes[int(i/2),i%2].axis('equal')
        axes[int(i/2),i%2].axis('off')
        
    plt.tight_layout()
    fig.savefig(fig7_dir + '/panel_7J.png', dpi=400)
    plt.close()
    
    
    ''' panel 7K '''
        
    all_normed_heatmaps = data_7JK['normed_resampled_heatmaps']

    heatmap_mins = []
    heatmap_maxes = []

    for cue_loc in ['cue_north','cue_south','cue_east','cue_west']:
        
        mean_mean_heatmaps = np.nanmean(all_normed_heatmaps[cue_loc],axis=0)        
        heatmap_mins.append(np.nanmin(mean_mean_heatmaps))
        heatmap_maxes.append(np.nanmax(mean_mean_heatmaps))
        
    heatmap_min = np.nanmin(heatmap_mins)
    heatmap_max = np.nanmax(heatmap_maxes)
    
    fig, axes = plt.subplots(2,2,figsize=(5,5))    

    for i in range(4):
        
        cue_loc = cue_locs[i]
        zone = zone_coords[cue_loc]

        mean_mean_heatmaps = np.nanmean(all_normed_heatmaps[cue_loc],axis=0)

        axes[int(i/2),i%2].imshow(mean_mean_heatmaps,vmin=heatmap_min,vmax=heatmap_max,origin='lower',cmap='viridis')
        axes[int(i/2),i%2].plot([zone[0]/zone_scale - .5,zone[1]/zone_scale - .5,zone[1]/zone_scale - .5,zone[0]/zone_scale - .5,zone[0]/zone_scale - .5],
                [zone[2]/zone_scale - .5,zone[2]/zone_scale - .5,zone[3]/zone_scale - .5,zone[3]/zone_scale - .5,zone[2]/zone_scale - .5],'r-',linewidth=3,alpha=.9)
        axes[int(i/2),i%2].axis('equal')
        axes[int(i/2),i%2].axis('off')
        
    plt.tight_layout()
    fig.savefig(fig7_dir + '/panel_7K.png', dpi=400)
    plt.close()
    
    
if __name__ == '__main__':
    make_figures()



