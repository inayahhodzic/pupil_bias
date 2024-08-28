# import relevant packages
import os
import numpy as np
import scipy as sp
from scipy import stats
from scipy.io import loadmat
import pandas as pd
import h5py

# import visualisation packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# import utilities
from IPython import embed as shell
import utils

# set figure specifications
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 6,
    'axes.linewidth': 0.25,
    'xtick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.width': 0.25,
    'ytick.major.pad' : 2.0,
    'ytick.minor.pad' : 2.0,
    'xtick.major.pad' : 2.0,
    'xtick.minor.pad' : 2.0,
    'axes.labelpad' : 4.0,
    'axes.titlepad' : 6.0,
    } )
sns.plotting_context()

def plot_pupil_bias(df, q=3):

    # binned data:
    res = (df.loc[df['trial_type']=='auditory',:].groupby(['subject_id', 'session_id', 'pupil_bin'])['correct'].mean()-
            df.loc[df['trial_type']=='visual',:].groupby(['subject_id', 'session_id', 'pupil_bin'])['correct'].mean()).reset_index()
    res['correct_abs'] = abs(res['correct'])
    for s, r in res.groupby(['session_id']):
        res.loc[r.index, 'correct_abs_dm'] = res.loc[r.index, 'correct_abs']-res.loc[r.index, 'correct_abs'].mean()

    # stats:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    vc = {'session_id': '0 + C(session_id)'}
    md = smf.mixedlm("correct_abs ~ pupil_bin",
                    vc_formula=vc, 
                    re_formula='1',
                    # re_formula='1 + {}'.format(x),
                    groups='subject_id', 
                    data=res,)
    mdf = md.fit()
    print(mdf.summary())

    # plot:
    means = res.groupby(['pupil_bin']).mean().reset_index()
    sems = res.groupby(['pupil_bin']).sem().reset_index()
    fig = plt.figure(figsize=(2,2))
    plt.errorbar(x=means['pupil_bin'], y=means['correct_abs'], yerr=sems['correct_abs_dm'], fmt='-o', color='darkgrey')
    plt.xlabel('Pupil (bin)')
    plt.ylabel('Modality bias)')
    sns.despine(trim=False)
    plt.tight_layout()

    return mdf, fig

def plot_pupil_physio(df, x='pupil_stim_b', y='early', plot=True):
    
    # stats:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    # vc = {'session_id': '0 + C(session_id)'}
    md = smf.mixedlm("{} ~ {}".format(y,x),
                    # vc_formula=vc, 
                    re_formula='1',
                    # re_formula='1 + {}'.format(x),
                    groups='session_id', 
                    data=df,)
    mdf = md.fit()
    print(mdf.summary())

    if plot:

        # binned data:
        res = df.groupby(['session_id', 'pupil_bin']).mean().reset_index()
        for s, r in res.groupby(['session_id']):
            res.loc[r.index, '{}_dm'.format(y)] = res.loc[r.index, y]-res.loc[r.index, y].mean()

        # plot:
        fontsize=9
        means = res.groupby(['pupil_bin']).mean().reset_index()
        sems = res.groupby(['pupil_bin']).sem().reset_index()
        fig = plt.figure(figsize=(2.5,2.5))
        plt.errorbar(x=means['pupil_bin'], y=means[y], yerr=sems['{}_dm'.format(y)], fmt='o', markersize=8, color='#841E52', linewidth=2, capsize=2)

        sns.regplot(x='pupil_bin', y=y, data=means, color='#77AC30', ci=None, scatter=False)

        plt.title('p = {}'.format(round(mdf.pvalues[x],3)), fontsize=fontsize)
        plt.xlabel('Pupil size (bin)', fontsize=fontsize, labelpad=7)
        plt.ylabel(y)
        sns.despine(trim=False)
        plt.tight_layout()

        return mdf, fig
    else:
        return mdf

def plot_timecourses(df, epochs, rt_cutoff=0.5):

    means = epochs.query("(area=='V1')").groupby(['session_id', 'trial_type', 'outcome']).mean().groupby(['trial_type', 'outcome']).mean()
    sems = epochs.query("(area=='V1')").groupby(['session_id', 'trial_type', 'outcome']).mean().groupby(['trial_type', 'outcome']).sem()

    fig = plt.figure(figsize=(6,3))

    ax = fig.add_subplot(121)
    plt.fill_between(means.columns, means.iloc[3]-sems.iloc[3], means.iloc[3]+sems.iloc[3], alpha=0.2)
    plt.plot(means.columns, means.iloc[3], label='miss')
    plt.fill_between(means.columns, means.iloc[2]-sems.iloc[2], means.iloc[2]+sems.iloc[2], alpha=0.2)
    plt.plot(means.columns, means.iloc[2], label='hit')
    plt.ylim(-0.15, 0.8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
    plt.xlabel('Time from change (s)', fontsize=10, labelpad=6)
    plt.ylabel('Spike rate (z)', fontsize=10, labelpad=6)
    plt.legend(fontsize=8)
    plt.axvline(0, color='darkgrey', linestyle='--')

    plt.axvspan(0,0.25, color='black', alpha=0.1)
    plt.axvspan(0.35,0.6, color='dimgrey', alpha=0.1)

    ax = fig.add_subplot(122)
    # plt.fill_between(means.columns, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], alpha=0.2)
    # plt.plot(means.columns, means.iloc[1])
    # plt.fill_between(means.columns, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], alpha=0.2)
    # plt.plot(means.columns, means.iloc[0])
    # plt.ylim(-0.1,0.8)

    d, e = utils.match_meta_epochs(df.loc[(df['rt']<rt_cutoff),:],
                                        epochs_s_stim.query("(area=='V1') & (trial_type == 'visual')"))
    means = e.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).mean()
    sems = e.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).sem()
    x = np.array(means.columns, dtype=float)
    plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color='grey', alpha=0.1)
    plt.plot(means.iloc[0], color='coral', linestyle='--', label='RT < 0.5s')
    # plt.axvline(d.groupby(['session_id', 'outcome'])['rt'].mean().groupby(['outcome']).mean().iloc[0])
    print(d.shape)

    d, e = utils.match_meta_epochs(df.loc[(df['rt']>rt_cutoff),:],
                                        epochs_s_stim.query("(area=='V1') & (trial_type == 'visual')"))
    means = e.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).mean()
    sems = e.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).sem()
    x = np.array(means.columns, dtype=float)
    plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], color=sns.color_palette()[1], alpha=0.1)
    plt.plot(means.iloc[0], color='sienna', label='RT > 0.5s')
    # plt.axvline(d.groupby(['session_id', 'outcome'])['rt'].mean().groupby(['outcome']).mean().iloc[0])
    plt.axvspan(0.35,0.6, color='dimgrey', alpha=0.1)
    print(d.shape)
    plt.legend(fontsize=8)
    plt.axvline(0, color='darkgrey', linestyle='--')
    plt.ylim(-0.15, 0.8)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8])
    plt.xlabel('Time from change (s)', fontsize=10, labelpad=6)
    plt.ylabel('Spike rate (z)', fontsize=10, labelpad=6)
    sns.despine(trim=False)
    plt.tight_layout()

    return fig

def compute_modality_bias(df, cutoff=0.15):

    correct_visual = df.loc[(df['trial_type']=='visual'), 'correct'].mean()
    correct_auditory = df.loc[(df['trial_type']=='auditory'), 'correct'].mean()
    modality_bias = (correct_visual - correct_auditory)
    performance = (correct_visual + correct_auditory)/2
    df['modality_bias'] = modality_bias
    return df

def plot_response(df, epochs_s_stim, measure='mean'):

    # match:
    df, epochs_s_stim = utils.match_meta_epochs(df, epochs_s_stim)

    # remove outliers:
    early = df.groupby(['subject_id', 'session_id', 'outcome'])['early'].mean().groupby(['subject_id', 'session_id']).mean().reset_index()
    early['z'] = (early['early'] - early['early'].mean()) / early['early'].std()
    early = early.loc[(early['z']>-2)&(early['z']<2),:]
    df = df.loc[df[['subject_id', 'session_id']].set_index(['subject_id', 'session_id']).index.isin(
                early.loc[:, ['subject_id', 'session_id']].set_index(['subject_id', 'session_id']).index),:]
    epochs_s_stim = epochs_s_stim.loc[epochs_s_stim.index.droplevel(list(range(2,13))).isin(early.loc[:, ['subject_id', 'session_id']].set_index(['subject_id', 'session_id']).index),:]

    # plot:
    plt_nr = 1
    fig = plt.figure(figsize=(6,2))
    for b, bias in zip([-0.15, 0.15], ['auditory', 'visual']):
        
        if b < 0:
            d, epochs = utils.match_meta_epochs(df.loc[df['modality_bias']<b,:], epochs_s_stim)
        elif b > 0:
            d, epochs = utils.match_meta_epochs(df.loc[df['modality_bias']>b,:], epochs_s_stim)

        # ind = epochs_s_stim.index.droplevel(list(range(2,12))).isin(df.loc[df['modality_bias']==b, ['subject_id', 'session_id']].set_index(['subject_id', 'session_id']).index)
        means = epochs.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).mean()
        sems = epochs.groupby(['session_id', 'outcome']).mean().groupby(['outcome']).sem()
        ax = fig.add_subplot(1,4,plt_nr)
        x = np.array(means.columns, dtype=float)
        plt.fill_between(x, means.iloc[0]-sems.iloc[0], means.iloc[0]+sems.iloc[0], alpha=0.2)
        plt.plot(x, means.iloc[0], label='hit')
        plt.fill_between(x, means.iloc[1]-sems.iloc[1], means.iloc[1]+sems.iloc[1], alpha=0.2)
        plt.plot(x, means.iloc[1], label='miss')
        plt.axvline(0, lw=0.5, color='k')
        plt.axvspan(0, 0.25, color='k', alpha=0.1)
        plt.ylim(-0.5, 1.5)
        plt.title('{} bias'.format(bias))
        plt.xlabel('Time from change (s)')
        plt.ylabel('Spike rate (z)')
        plt.legend()
        plt_nr+=1

    # ax = fig.add_subplot(1,3,plt_nr)
    # bias = df.groupby(['session_id'])['modality_bias'].mean()
    # means = epochs_s_stim.groupby(['session_id', 'outcome']).mean().groupby(['session_id']).mean()
    # x = means.columns
    # early = means.loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    # r,p = sp.stats.pearsonr(bias, early[0])
    # sns.regplot(x=early[0], y=bias)
    # plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
    # plt.xlabel('Spike rate (z)')
    # plt.ylabel('Modality bias')
    # plt_nr+=1

    ax = fig.add_subplot(1,4,plt_nr)
    bias = df.groupby(['session_id'])['modality_bias'].mean()
    if measure == 'mean':
        early = df.groupby(['session_id', 'outcome'])['early'].mean().groupby('session_id').mean()
    elif measure == 'diff':
        early = df.groupby(['session_id', 'outcome'])['early'].mean().groupby('session_id').diff().loc[::2]
    
    r,p = sp.stats.pearsonr(bias, early)
    sns.regplot(x=early, y=bias)
    plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
    plt.xlabel('Spike rate (z)')
    plt.ylabel('Modality bias')
    plt_nr+=1

    ax = fig.add_subplot(1,4,plt_nr)
    bias = df.groupby(['session_id'])['modality_bias'].mean().reset_index()
    if measure == 'mean':
        # late = df.groupby(['session_id', 'outcome'])['late'].mean().groupby('session_id').mean().reset_index()
        late = df.loc[df['outcome']==0,:].groupby(['session_id'])['late'].mean().reset_index()
    elif measure == 'diff':
        late = df.groupby(['session_id', 'outcome'])['late'].mean().groupby('session_id').diff()
    res = bias.merge(late, on='session_id')
    r,p = sp.stats.pearsonr(res['modality_bias'], res['late'])
    sns.regplot(x=res['late'], y=res['modality_bias'])
    plt.title('r = {}, p = {}'.format(round(r,2), round(p,3)))
    plt.xlabel('Spike rate (z)')
    plt.ylabel('Modality bias')

    sns.despine(trim=False)
    plt.tight_layout()
    return fig

def compute_moving_averages(df, window=25):

    slice_window = (window-1)/2

    df['moving_correct'] = np.NaN
    df['moving_bias'] = np.NaN
    for t in df['trial_id']:
        
        ind_auditory = ((df['trial_type']=='auditory') &
                                        (df['trial_id']>=(t-slice_window)) & 
                                        (df['trial_id']<=(t+slice_window)))
        ind_visual = ((df['trial_type']=='visual') &
                                        (df['trial_id']>=(t-slice_window)) & 
                                        (df['trial_id']<=(t+slice_window)))
        if t < 10:
            continue
        elif t > (max(df['trial_id'])-10):
            continue
        elif (sum(ind_auditory) < 5) | (sum(ind_visual) < 5):
            continue
        else:
            moving_correct_auditory = df.loc[ind_auditory, 'correct'].mean()
            moving_correct_visual = df.loc[ind_visual, 'correct'].mean()

            df.loc[df['trial_id']==t, 'moving_correct_visual'] = moving_correct_visual
            df.loc[df['trial_id']==t, 'moving_correct_auditory'] = moving_correct_auditory
             
            df.loc[df['trial_id']==t, 'moving_correct'] = (moving_correct_visual+moving_correct_auditory)/2
            df.loc[df['trial_id']==t, 'moving_bias'] = (moving_correct_visual-moving_correct_auditory)
    return df

# load:
prepare = 0
n_jobs = 12
# n_jobs = 1
raw_dir = 'data/CHDET/'
if prepare:
    (df, epochs_p_stim, epochs_p_lick, epochs_s_stim, 
        epochs_s_lick, epochs_s_stim_z, epochs_s_lick_z) = utils.load_data(raw_dir=raw_dir, n_jobs=n_jobs)
    df.to_csv('data/df.csv')
    epochs_p_stim.to_hdf('data/epochs_p_stim.hdf', key='pupil')
    epochs_p_lick.to_hdf('data/epochs_p_lick.hdf', key='pupil')
    epochs_s_stim.to_hdf('data/epochs_s_stim.hdf', key='spike')
    epochs_s_lick.to_hdf('data/epochs_s_lick.hdf', key='spike')
    epochs_s_stim_z.to_hdf('data/epochs_s_stim_z.hdf', key='spike')
    epochs_s_lick_z.to_hdf('data/epochs_s_lick_z.hdf', key='spike')

# for data_split in ['all', 'pupil', 'spike', 'pupil_spike']:
for data_split in ['pupil_spike']:
# for data_split in ['spike']:

    # load:
    df = pd.read_csv('data/df.csv')
    if 'pupil' in data_split:
        epochs_p_stim = pd.read_hdf('data/epochs_p_stim.hdf', key='pupil')
        epochs_p_lick = pd.read_hdf('data/epochs_p_lick.hdf', key='pupil')
    if 'spike' in data_split:
        epochs_s_stim = pd.read_hdf('data/epochs_s_stim.hdf', key='spike')
        epochs_s_stim_z = pd.read_hdf('data/epochs_s_stim_z.hdf', key='spike')
        epochs_s_lick = pd.read_hdf('data/epochs_s_lick.hdf', key='spike')
        epochs_s_lick_z = pd.read_hdf('data/epochs_s_lick_z.hdf', key='spike')

    df['fa_rate'] = df['nr_lick_bouts'] / (df['time_trial_end']-df['time_trial_start'])
    df['difficulty'] = np.NaN
    df.loc[df['trial_type']=='visual', 'difficulty'] = df.loc[df['trial_type']=='visual', 'visualOriChangeNorm']
    df.loc[df['trial_type']=='auditory', 'difficulty'] = df.loc[df['trial_type']=='auditory', 'audioFreqChangeNorm']
    df['trial_dur'] = df['time_trial_end']-df['time_trial_start']
    df['stimulus_p'] = (df['outcome_p']==0)|(df['outcome_p']==1)
    df['choice_p'] = (df['outcome_p']==0)|(df['outcome_p']==2)
    df['choice'] = (df['outcome']==0)|(df['outcome']==2)
    
    df = utils.select_data_behavior(df)
    if data_split == 'all':
        pass
    elif data_split == 'pupil':
        df, epochs_p_stim, epochs_p_lick = utils.select_data_pupil(df.copy(), epochs_p_stim.copy(), epochs_p_lick.copy())
    
    elif data_split == 'spike':
        df, epochs_s_stim, epochs_s_lick = utils.select_data_spike(df.copy(), epochs_s_stim.copy(), epochs_s_lick.copy())
    elif data_split == 'pupil_spike':
        df, epochs_p_stim, epochs_p_lick = utils.select_data_pupil(df.copy(), epochs_p_stim.copy(), epochs_p_lick.copy())
        df, epochs_s_stim, epochs_s_lick = utils.select_data_spike(df.copy(), epochs_s_stim.copy(), epochs_s_lick.copy())

    epochs_s_stim.replace([np.inf, -np.inf], np.nan, inplace=True)
    epochs_s_lick.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # remove outliers:
    # epochs_s_stim = epochs_s_stim.query("(fr_avg<30)")
    columns = ['subject_id', 'session_id', 'cell_id']
    x = epochs_s_stim.columns
    resp = epochs_s_stim.query("(area=='V1') & (trial_type == 'visual')").groupby(columns).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    resp = resp.rename({0:'early_activity'}, axis=1)
    # plt.hist(resp['early_activity'], bins=100)
    resp = resp.loc[(resp['early_activity']<4),:]
    epochs_s_stim = epochs_s_stim.loc[epochs_s_stim.index.get_level_values('cell_id').isin(resp['cell_id'])]
  
    # add bias:
    df = df.groupby(['subject_id', 'session_id'], group_keys=False).apply(compute_modality_bias)

    # add early and late V1 activity
    x = epochs_s_stim.columns

    baseline = epochs_s_stim.query("(area=='V1')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>-0.35)&(x<-0.1)].mean(axis=1).reset_index()
    baseline.columns = ['subject_id', 'session_id', 'trial_id', 'baseline']
    df = df.merge(baseline, on=['subject_id', 'session_id', 'trial_id'], how='left')

    early = epochs_s_stim.query("(area=='V1')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early.columns = ['subject_id', 'session_id', 'trial_id', 'early']
    df = df.merge(early, on=['subject_id', 'session_id', 'trial_id'], how='left')

    late = epochs_s_stim.query("(area=='V1')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.5)&(x<1)].mean(axis=1).reset_index()
    late.columns = ['subject_id', 'session_id', 'trial_id', 'late']
    df = df.merge(late, on=['subject_id', 'session_id', 'trial_id'], how='left')

    late = epochs_s_stim.query("(area=='V1')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late.columns = ['subject_id', 'session_id', 'trial_id', 'late2']
    df = df.merge(late, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['early'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['late'].isna(),:].reset_index(drop=True)

    # df['early'] = df['early'] - df['baseline']
    # df['late'] = df['late'] - df['baseline']
    # df['late2'] = df['late2'] - df['baseline']
    
    #baseline activity and layers
    baseline_SG = epochs_s_stim.query("(area=='V1')&(layer=='SG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>-0.35)&(x<-0.1)].mean(axis=1).reset_index()
    baseline_SG.columns = ['subject_id', 'session_id', 'trial_id', 'baseline_SG']
    baseline_G = epochs_s_stim.query("(area=='V1')&(layer=='G')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>-0.35)&(x<-0.1)].mean(axis=1).reset_index()
    baseline_G.columns = ['subject_id', 'session_id', 'trial_id', 'baseline_G']
    baseline_IG = epochs_s_stim.query("(area=='V1')&(layer=='IG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>-0.35)&(x<-0.1)].mean(axis=1).reset_index()
    baseline_IG.columns = ['subject_id', 'session_id', 'trial_id', 'baseline_IG']

    df = df.merge(baseline_SG, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(baseline_G, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(baseline_IG, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['baseline_SG'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['baseline_G'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['baseline_IG'].isna(),:].reset_index(drop=True)

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='baseline_SG')
    fig.savefig('figs/pupil_layers_baselineSG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='baseline_G')
    fig.savefig('figs/pupil_layers_baselineG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='baseline_IG')
    fig.savefig('figs/pupil_layers_baselineIG_{}.pdf'.format(data_split))

    # early activity and layers
    early_SG = epochs_s_stim.query("(area=='V1')&(layer=='SG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_SG.columns = ['subject_id', 'session_id', 'trial_id', 'early_SG']
    early_G = epochs_s_stim.query("(area=='V1')&(layer=='G')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_G.columns = ['subject_id', 'session_id', 'trial_id', 'early_G']
    early_IG = epochs_s_stim.query("(area=='V1')&(layer=='IG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_IG.columns = ['subject_id', 'session_id', 'trial_id', 'early_IG']

    df = df.merge(early_SG, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(early_G, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(early_IG, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['early_SG'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['early_G'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['early_IG'].isna(),:].reset_index(drop=True)

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_SG')
    fig.savefig('figs/pupil_layers_earlySG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_G')
    fig.savefig('figs/pupil_layers_earlyG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_IG')
    fig.savefig('figs/pupil_layers_earlyIG_{}.pdf'.format(data_split))

    # late activity and layers
    late_SG = epochs_s_stim.query("(area=='V1')&(layer=='SG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_SG.columns = ['subject_id', 'session_id', 'trial_id', 'late_SG']
    late_G = epochs_s_stim.query("(area=='V1')&(layer=='G')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_G.columns = ['subject_id', 'session_id', 'trial_id', 'late_G']
    late_IG = epochs_s_stim.query("(area=='V1')&(layer=='IG')").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_IG.columns = ['subject_id', 'session_id', 'trial_id', 'late_IG']

    df = df.merge(late_SG, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(late_G, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(late_IG, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['late_SG'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['late_G'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['late_IG'].isna(),:].reset_index(drop=True)

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_SG')
    fig.savefig('figs/pupil_layers_lateSG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_G')
    fig.savefig('figs/pupil_layers_lateG_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_IG')
    fig.savefig('figs/pupil_layers_lateIG_{}.pdf'.format(data_split))

    # cell_types, early
    early_1 = epochs_s_stim.query("(area=='V1')&(cell_type==0)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_1.columns = ['subject_id', 'session_id', 'trial_id', 'early_1']
    early_2 = epochs_s_stim.query("(area=='V1')&(cell_type==1)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_2.columns = ['subject_id', 'session_id', 'trial_id', 'early_2']
    early_3 = epochs_s_stim.query("(area=='V1')&(cell_type==2)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    early_3.columns = ['subject_id', 'session_id', 'trial_id', 'early_3']

    df = df.merge(early_1, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(early_2, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(early_3, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['early_1'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['early_2'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['early_3'].isna(),:].reset_index(drop=True)

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_1')
    fig.savefig('figs/pupil_layers_early1_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_2')
    fig.savefig('figs/pupil_layers_early2_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early_3')
    fig.savefig('figs/pupil_layers_early3_{}.pdf'.format(data_split))

    # cell_types, late
    late_1 = epochs_s_stim.query("(area=='V1')&(cell_type==0)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_1.columns = ['subject_id', 'session_id', 'trial_id', 'late_1']
    late_2 = epochs_s_stim.query("(area=='V1')&(cell_type==1)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_2.columns = ['subject_id', 'session_id', 'trial_id', 'late_2']
    late_3 = epochs_s_stim.query("(area=='V1')&(cell_type==2)").groupby(['subject_id', 'session_id', 'trial_id']).mean().loc[:,(x>0.35)&(x<0.6)].mean(axis=1).reset_index()
    late_3.columns = ['subject_id', 'session_id', 'trial_id', 'late_3']

    df = df.merge(late_1, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(late_2, on=['subject_id', 'session_id', 'trial_id'], how='left')
    df = df.merge(late_3, on=['subject_id', 'session_id', 'trial_id'], how='left')

    df = df.loc[~df['late_1'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['late_2'].isna(),:].reset_index(drop=True)
    df = df.loc[~df['late_3'].isna(),:].reset_index(drop=True)

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_1')
    fig.savefig('figs/pupil_layers_late1_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_2')
    fig.savefig('figs/pupil_layers_late2_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late_3')
    fig.savefig('figs/pupil_layers_late3_{}.pdf'.format(data_split))

    # # df.loc[df['trial_type']=='visual',:].groupby(['subject_id', 'session_id', 'outcome']).mean()
    # df = df.groupby(['subject_id', 'session_id']).apply(compute_moving_averages)
    # df = df.loc[~df['moving_correct'].isna(),:]
    
    # df['nr_trials_total'] = df.groupby(['session_id'])['trial_id'].transform('count')
    # df = df.loc[df['nr_trials_total']>100,:]

    def trial_selection(df):
        return df.loc[(df['trial_id']>10)&(df['trial_id']<(df['trial_id'].max()-10))]
    df = df.groupby(['session_id']).apply(trial_selection).reset_index(drop=True)

    # fig = plot_response(df, epochs_s_stim.query("(area=='V1') & (trial_type == 'visual')"))
    # fig.savefig('figs/spikes_early_{}.pdf'.format(data_split))

    # add pupil bins:
    q = 5
    df['pupil_bin'] = df.groupby(['session_id', 'trial_type'])['pupil_stim_b'].apply(pd.qcut, q=q, labels=False, duplicates='drop')

    # plot relationship between pupil and modality bias:
    mdf, fig = plot_pupil_bias(df)
    fig.savefig('figs/pupil_bias_{}.pdf'.format(data_split))

    # for s, d in df.groupby(['session_id']):
    #     mdf, fig = plot_pupil_bias(d)

    # add pupil bins:
    q = 4
    df['pupil_bin'] = np.NaN
    df['pupil_bin'] = df.groupby(['session_id', 'trial_type'])['pupil_stim_b'].apply(pd.qcut, q=q, labels=False, duplicates='drop')
    # df.loc[df['outcome']==0,'pupil_bin'] = df.loc[df['outcome']==0,:].groupby(['session_id', 'trial_type'])['pupil_lick'].apply(pd.qcut, q=q, labels=False, duplicates='drop')

    # plot timecourses:
    fig = plot_timecourses(df, epochs_s_stim, rt_cutoff=0.5)
    fig.savefig('figs/timecourses_{}.pdf'.format(data_split))

    # plot relationship between pupil and physio:

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='baseline')
    fig.savefig('figs/pupil_baseline_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early')
    fig.savefig('figs/pupil_early_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0),:], x='pupil_stim_b', y='late')
    fig.savefig('figs/pupil_late_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0),:], x='pupil_stim_b', y='late2')
    fig.savefig('figs/pupil_late2_{}.pdf'.format(data_split))

    mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_stim_b', y='late2')
    fig.savefig('figs/pupil_late2_{}.pdf'.format(data_split))
    
    #make overall figure
    fig = plt.figure(figsize=(6,6))
    fig.add_subplot(121)
    plot_pupil_physio(df.loc[(df['trial_type']=='visual'),:], x='pupil_stim_b', y='early')

    fig.add_subplot(122)
    plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0),:], x='pupil_stim_b', y='late2')
    plt.tight_layout()

    # mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_lick', y='early')
    # mdf, fig = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=0.4),:], x='pupil_lick', y='late2')

    # plot as function of RT
    fontsize=9.5
    rts = np.arange(0,1.1,0.05)
    ts = np.zeros(len(rts))
    ps = np.zeros(len(rts))
    trials = np.zeros(len(rts))
    for i, rt in enumerate(rts):
        mdf = plot_pupil_physio(df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=rt),:], x='pupil_stim_b', y='late2', plot=False)
        ts[i] = mdf.tvalues['pupil_stim_b']
        ps[i] = mdf.pvalues['pupil_stim_b']
        trials[i] = df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>=rt),:].shape[0]
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(311)
    plt.plot(rts, ts, linewidth=2, color='#841E52')
    plt.ylabel('T-statistic', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = fig.add_subplot(312)
    plt.plot(rts, np.log10(ps), linewidth=2, color='#841E52')
    plt.axhline(np.log10(0.05), ls='--', color='dimgrey', linewidth=1)
    plt.ylabel('p-value', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax = fig.add_subplot(313)
    plt.plot(rts, trials, linewidth=2, color='#841E52')
    plt.xlabel('Exclude trials with RT >', fontsize=fontsize, labelpad=7)
    plt.ylabel('Total # of hits', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig('figs/rt_control_{}.pdf'.format(data_split))

    # statistical testing pupil baseline
    import scipy.stats as stats

    x = epochs_p_stim.columns
    means = epochs_p_stim.groupby(['subject_id', 'outcome']).mean().groupby(['outcome']).mean().loc[:,(x>-0.5)&(x<0)]
    means.reset_index()

    t_statistic, p_value = stats.ttest_ind(group0, group1)

    # df['pupil_bin'] = df.groupby(['session_id', 'trial_type'])['pupil_stim_b'].apply(pd.qcut, q=8, labels=False, duplicates='drop')
    # res = df.loc[(df['trial_type']=='visual')&~df['early'].isna(),:].groupby(['session_id', 'pupil_bin'])['early'].mean().reset_index()

    # for s, r in res.groupby(['session_id']):
    #     res.loc[r.index, 'early_dm'] = res.loc[r.index, 'early']-res.loc[r.index, 'early'].mean()
    # means = res.groupby(['pupil_bin']).mean().reset_index()
    # sems = res.groupby(['pupil_bin']).sem().reset_index()

    # plt.figure()
    # plt.errorbar(x=means['pupil_bin'], y=means['early'], yerr=sems['early_dm'])

    # y = 'late2'
    # df['pupil_bin'] = df.groupby(['session_id', 'trial_type'])['pupil_stim_b'].apply(pd.qcut, q=q, labels=False, duplicates='drop')
    # res = df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>0.4),:].groupby(['session_id', 'pupil_bin']).mean()['late2'].reset_index()
    # print(res)

    # res = df.loc[(df['trial_type']=='visual')&(df['outcome']==0)&(df['rt']>0.4),:].groupby(['session_id', 'pupil_bin'])['late2'].mean().reset_index()
    # print(res)

    # for s, r in res.groupby(['session_id']):
    #     res.loc[r.index, 'late2_dm'] = res.loc[r.index, 'late2']-res.loc[r.index, 'late2'].mean()
    # means = res.groupby(['pupil_bin']).mean().reset_index()
    # sems = res.groupby(['pupil_bin']).sem().reset_index()

    # plt.figure()
    # plt.errorbar(x=means['pupil_bin'], y=means['late2'], yerr=sems['late2_dm'])

    # plt.figure()
    # sns.pointplot(x='pupil_bin', y='late2',errorbar='se', data=res)
    
    # sns.pointplot(x='pupil_bin', y='early', hue='trial_type', errorbar='se', data=res)

    # linear_model(df, y='moving_bias')
    # linear_model(df, y='moving_correct')

    # # linear_model(df.loc[(~df['early'].isna())&df['outcome']==0,:], y='early', x='rt', q=3)
    # linear_model(df.loc[~df['early'].isna(),:], x='correct', y='early', q=2)

    # linear_model(df.loc[df['outcome']==0,:], y='rt', early_measure='late', q=3)

    # vc = {'session_id': '0 + C(session_id)'}
    # md = smf.mixedlm("correct ~ early * trial_type * pupil_stim_b",
    #                     vc_formula=vc, 
    #                     re_formula='1',
    #                     groups='subject_id', 
    #                     data=df.loc[:,:],)
    # mdf = md.fit()
    # print(mdf.summary())
 
    ax= fig.add_subplot(121)

    window=25
    slice_window = (window-1)/2
    df['moving_correct'] = np.NaN
    df['moving_bias'] = np.NaN
    df['moving_correct_vis'] = np.NaN
    df['moving_correct_aud'] = np.NaN 
    
    for t in df['trial_id']:
        ind_auditory = ((df['trial_type']=='auditory') &
                                        (df['trial_id']>=(t-slice_window)) &
                                        (df['trial_id']<=(t+slice_window)))
        ind_visual = ((df['trial_type']=='visual') &
                                        (df['trial_id']>=(t-slice_window)) &
                                        (df['trial_id']<=(t+slice_window)))
        if t < 10:
            continue
        elif t > (max(df['trial_id'])-10):
            continue
        elif (sum(ind_auditory) < 5) | (sum(ind_visual) < 5):
            continue
        else:
            moving_correct_auditory = df.loc[ind_auditory, 'correct'].mean()
            moving_correct_visual = df.loc[ind_visual, 'correct'].mean()
            df.loc[df['trial_id']==t, 'moving_correct'] = (moving_correct_visual+moving_correct_auditory)/2
            df.loc[df['trial_id']==t, 'moving_bias'] = (moving_correct_visual-moving_correct_auditory)
            df.loc[df['trial_id']==t, 'moving_correct_vis'] = moving_correct_visual
            df.loc[df['trial_id']==t, 'moving_correct_aud'] = moving_correct_auditory


    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(3,4))
    grid = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

    sessions = list(df['session_id'].unique())
    sns.set_palette('flare')
    colors = sns.color_palette('flare')
    fontsize = 9
    saved_avg = []

    plt = fig.add_subplot(grid[0])
    for ses in sessions:
        data = df.loc[df['session_id']==ses]
        average = data['moving_correct_vis'].mean() 
        # plt.scatter(1, average, color=colors[3], s=26)
        # plt.ylim(0.425, 0.6)
        # plt.xlim(0.94, 1.04)
        # plt.xticks([])
        # plt.ylabel('Average accuracy',  fontsize=fontsize)
        # plt.yticks(fontsize=fontsize)
        saved_avg.append(average)
    # calculate statistics
    overall_median = np.median(saved_avg)
    lower_quart = np.percentile(saved_avg, 25)
    upper_quart = np.percentile(saved_avg, 75)

    #overall_median = df['moving_correct_vis'].median()
    #lower_quart = df['moving_correct_vis'].quantile(0.25)
    #upper_quart = df['moving_correct_vis'].quantile(0.75)

    error = np.array([[overall_median - lower_quart], [upper_quart - overall_median]])

    plt.errorbar(0.98, overall_median, yerr=error, capsize=5, color='dimgrey', linewidth=2)
    plt.plot(0.98, overall_median, 'ro', color='dimgrey')
    plt.title('Average accuracy per session \n on visual hit trials', fontsize=fontsize)
    plt.tight_layout()

    # plot reaction time of visual trials
    ax2 = fig.add_subplot(grid[1])
    colors = sns.color_palette('flare')
    fontsize=9
    plt.hist(df.loc[df['trial_type']=='visual', 'rt'], bins=20, color=colors[-1])
    plt.axvline(0.525, color='dimgray', linestyle='--')
    plt.xlabel('Reaction time (s)', fontsize=fontsize, labelpad=8)
    plt.ylabel('Number of trials', fontsize=fontsize, labelpad=8)
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.title('Distribution of reaction times: Insights from 1200 visual trials', fontsize=10, pad=10)

    # two figures together
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    fig = plt.figure(figsize=(8, 4))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 4])
    plt.tight_layout()

    sessions = list(df['session_id'].unique())
    sns.set_palette('flare')
    colors = sns.color_palette('flare')
    fontsize = 10
    saved_avg = []

    ax1 = fig.add_subplot(grid[0])
    for ses in sessions:
        data = df.loc[df['session_id'] == ses]
        average = data['moving_correct_vis'].mean()
        ax1.scatter(1, average, color=colors[3], s=26)
        ax1.set_ylim(0.4, 0.6)
        ax1.set_yticks([0.40, 0.44, 0.48, 0.52, 0.56, 0.60])
        ax1.set_xlim(0.94, 1.04)
        ax1.set_xticks([])
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)   
        ax1.set_ylabel('Average accuracy', fontsize=fontsize, labelpad=8)
        ax1.tick_params(axis='y', labelsize=fontsize)
        saved_avg.append(average)
    overall_median = np.median(saved_avg)
    lower_quart = np.percentile(saved_avg, 25)
    upper_quart = np.percentile(saved_avg, 75)

    error = np.array([[overall_median - lower_quart], [upper_quart - overall_median]])
    error = np.array([[overall_median - lower_quart], [upper_quart - overall_median]])
    plt.errorbar(0.98, overall_median, yerr=error, capsize=5, color='dimgrey', linewidth=2)
    ax1.plot(0.98, overall_median, 'ro', color='dimgrey')
    ax1.set_title('Average accuracy per session \n on visual trials', fontsize=fontsize)
    plt.tight_layout()

    ax2 = fig.add_subplot(grid[1])
    colors = sns.color_palette('flare')
    ax2.hist(df.loc[df['trial_type'] == 'visual', 'rt'], bins=20, color=colors[-1])
    ax2.axvline(0.525, color='dimgray', linestyle='--')
    ax2.set_xlabel('Reaction time (s)', fontsize=fontsize, labelpad=8)
    ax2.set_ylabel('Number of trials', fontsize=fontsize, labelpad=8)
    ax2.tick_params(axis='both', labelsize=fontsize)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_title('Distribution of reaction times: Insights from 1200 visual trials', fontsize=fontsize, pad=10, fontfamily='Arial')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3)

