# import relevant packages
import os
import numpy as np
import scipy as sp
from scipy import stats
from scipy.io import loadmat
import pandas as pd
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm

# import visualisation packages
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from IPython import embed as shell

# set visualisation specifications
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.set(style='ticks', font='Arial', font_scale=1, rc={
    'axes.labelsize': 7,
    'axes.titlesize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
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

def gauss(x, a, mu, sig):
    return a * np.exp(-1.0 * ((x - mu)**2.0) / (2.0 * sig**2.0))

def compute_nr_lick_bouts(df):
    # import warnings
    # warnings.filterwarnings("error")
    if isinstance(df['lickTime'], list) & ~np.isnan(df['respwinStart']):
        lick_times = np.array(df['lickTime'])
        lick_times = lick_times[(lick_times<df['respwinStart'])]
        return (np.diff(lick_times/1000000) > 1).sum() + 1
    elif isinstance(df['lickTime'], list) & np.isnan(df['respwinStart']):
        lick_times = np.array(df['lickTime'])
        return (np.diff(lick_times/1000000) > 1).sum() + 1
    else:
        return np.NaN

def make_epochs(df, df_meta, locking, start, dur, measure, fs, baseline=False, b_start=-1, b_dur=1):

    # make sure we start with index 0:
    df_meta = df_meta.reset_index(drop=True)

    # locking_inds = np.array(df['time'].searchsorted(df_meta.loc[~df_meta[locking].isna(), locking]).ravel())
    locking_inds = np.array(df['time'].searchsorted(df_meta[locking]).ravel())
    # locking_inds = np.array([find_nearest(np.array(df['time']), t) for t in df_meta[locking]])

    start_inds = locking_inds + int(start/(1/fs))
    end_inds = start_inds + int(dur/(1/fs)) - 1
    start_inds_b = locking_inds + int(b_start/(1/fs))
    end_inds_b = start_inds_b + int(b_dur/(1/fs))
    
    epochs = []
    for s, e, sb, eb in zip(start_inds, end_inds, start_inds_b, end_inds_b):
        epoch = np.array(df.loc[s:e, measure]) 
        if baseline:
            epoch = epoch - np.array(df.loc[sb:eb,measure]).mean()
        if s < 0:
            epoch = np.concatenate((np.repeat(np.NaN, abs(s)), epoch))
        epochs.append(epoch)
    epochs = pd.DataFrame(epochs)
    epochs.columns = np.arange(start, start+dur, 1/fs).round(5)
    if df_meta[locking].isna().sum() > 0:
        epochs.loc[df_meta[locking].isna(),:] = np.NaN

    return epochs

def load_data(raw_dir, n_jobs=24):

    all_sessions = []
    # for exp in ['BehaviorConflict', 'ChangeDetectionConflictDecor', 'VisOnlyPsychophysics']:
    for exp in ['ChangeDetectionConflict', 'BehaviorConflict']:
    # for exp in ['ChangeDetectionConflict']:
        subjects = os.listdir(os.path.join(raw_dir, exp))
        for subj in subjects:
            if subj == '.DS_Store':
                continue
            sessions = os.listdir(os.path.join(raw_dir, exp, subj))
            for ses in sessions:
                if not ( os.path.exists(os.path.join(raw_dir, exp, subj, ses, 'sessionData.mat')) &
                         os.path.exists(os.path.join(raw_dir, exp, subj, ses, 'trialData.mat'))):
                    continue
                all_sessions.append((raw_dir, exp, subj, ses))
    
    # run:
    backend = 'loky'
    # load_data_session(raw_dir, 'ChangeDetectionConflict', '2026', '2020-01-21_12-50-34')
    res = Parallel(n_jobs=n_jobs, verbose=1, backend=backend)(delayed(load_data_session)(*session)
                                                                               for session in tqdm(all_sessions))

    # unpack:
    df = pd.concat(res[i][0] for i in range(len(res)))
    epochs_p_stim = pd.concat(res[i][1] for i in range(len(res)))
    epochs_p_lick = pd.concat(res[i][2] for i in range(len(res)))
    epochs_s_stim = pd.concat(res[i][3] for i in range(len(res)))
    epochs_s_lick = pd.concat(res[i][4] for i in range(len(res)))
    epochs_s_stim_z = pd.concat(res[i][5] for i in range(len(res)))
    epochs_s_lick_z = pd.concat(res[i][6] for i in range(len(res)))

    return df, epochs_p_stim, epochs_p_lick, epochs_s_stim, epochs_s_lick, epochs_s_stim_z, epochs_s_lick_z

def load_data_session(raw_dir, exp, subj, ses):
    
    micro_to_s = 1000000
    bin_width = 0.01 # in seconds
    sigma = 0.05     # in seconds
    win_len = 0.3

    # load session data:
    session_mat = loadmat(os.path.join(raw_dir, exp, subj, ses, 'sessionData.mat'))['sessionData']
    data = [[row.flat[0] for row in line] for line in session_mat[0][0]]
    df_session = pd.DataFrame(data).T
    df_session.columns = session_mat.dtype.names
    # print(df_session.columns)
    columns = ['mousename', 'session_ID', 'Genotype', 'Rec_datetime', 'Experiment', 'Parameterfile', 'Protocolfile']
    df_session = df_session[columns]
    df_session = df_session.iloc[0:1]
    df_session = df_session.applymap(lambda x: x[0] if (type(x)==np.ndarray) else x)

    # load trial data:
    trial_mat = loadmat(os.path.join(raw_dir, exp, subj, ses, 'trialData.mat'))['trialData']
    data = [[row.flat[0] for row in line] for line in trial_mat[0][0]]
    df_trial = pd.DataFrame(data).T
    df_trial.columns = trial_mat.dtype.names
    # print(df_trial.columns)

    columns = ['session_ID', 'trialType', 'trialNum', 'trialStart', 'trialEnd', 'stimChange',
                'respwinStart',
                'visualOriChangeNorm', 'audioFreqChangeNorm', 'responseLatency',
                'leftCorrect', 'rightCorrect', 'lickTime', 'lickSide', 'responseSide', 
                'correctResponse', 'noResponse', 'rewardTime', 'rewardSide', 
                'vecResponse', 'hasvisualchange', 'hasaudiochange',]
    df_trial = df_trial[columns]
    df_trial = df_trial.applymap(lambda x: list(x) if (type(x)==np.ndarray) else x)
    df_trial = df_trial.applymap(lambda x: np.nan if x==[] else x)
    df_trial = df_trial.applymap(lambda x: x[0] if (type(x)==list) else x)
    df_trial = df_trial.applymap(lambda x: list(x) if (type(x)==np.ndarray) else x)
    df_trial = df_trial.applymap(lambda x: np.nan if x==[] else x)

    # merge:
    df = df_session.merge(df_trial, on='session_ID')
    
    # rename:
    df.loc[(df['trialType']=='P'), 'stimChange'] = np.NaN
    df['nr_licks'] = df['lickTime'].str.len()
    df['nr_lick_bouts'] = df.apply(compute_nr_lick_bouts, axis=1)
    df = df.rename({'Experiment': 'experiment', 
                    'trialType':'trial_type', 
                    'mousename': 'subject_id',
                    'session_ID': 'session_id',
                    'trialNum': 'trial_id',
                    'trialStart': 'time_trial_start',
                    'trialEnd': 'time_trial_end',
                    'stimChange': 'time_stim',
                    'responseLatency': 'rt',
                    'correctResponse': 'correct'}, axis=1)
    # df['experiment'] = df['experiment'].map({
    #                                         'BehaviorConflict': 'MST', 
    #                                         'ChangeDetectionConflictDecor': 'NE', 
    #                                         'VisOnlyPsychophysics': 'UST'
    #                                         })
    df['trial_type'] = df['trial_type'].map({'X': 'visual', 'Y': 'auditory', 'P': 'probe', 'C':'conflict'})
    df['outcome'] = np.NaN
    df.loc[(df['correct']==1) & (df['trial_type']!='probe'), 'outcome'] = 0
    df.loc[(df['correct']==0) & (df['trial_type']!='probe'), 'outcome'] = 1
    df.loc[(df['correct']==0) & (df['trial_type']=='probe'), 'outcome'] = 2
    df.loc[(df['correct']==1) & (df['trial_type']=='probe'), 'outcome'] = 3
    df['outcome_p'] = np.NaN
    df.loc[df['trial_id'].diff()==1, 'outcome_p'] = df['outcome'].iloc[np.where(df['trial_id'].diff()==1)[0]-1].values

    # lick times:
    ind = (df['correct']==1)&(df['trial_type']!='probe')
    df['time_lick'] = np.NaN
    df.loc[ind, 'time_lick'] = (df.loc[ind, 'time_stim'] + df.loc[ind, 'rt'])
    
    # to seconds:
    df['time_trial_start'] = df['time_trial_start'] / micro_to_s
    df['time_trial_end'] = df['time_trial_end'] / micro_to_s
    df['time_stim'] = df['time_stim'] / micro_to_s
    df['rt'] = df['rt'] / micro_to_s
    df['time_lick'] = df['time_lick'] / micro_to_s
    df['latency_stim'] = df['time_stim'] - df['time_trial_start']
    df['latency_lick'] = df['time_lick'] - df['time_trial_start']

    if os.path.exists(os.path.join(raw_dir, exp, subj, ses, 'videoData.mat')):

        # load video data:
        f = h5py.File(os.path.join(raw_dir, exp, subj, ses, 'videoData.mat'))
        video_data = pd.DataFrame({
            'time': np.array(f[f['/videoData/ts'][0][0]])[0] / micro_to_s,
            'area': np.array(f[f['/videoData/area'][0][0]])[0],
            'int': np.array(f[f['/videoData/interp'][0][0]])[0],
            'isgood': np.array(f[f['/videoData/isgood'][0][0]])[0]
        })

        video_data['area_psc'] = (video_data['area']-video_data['area'].median()) / video_data['area'].median() * 100
        video_data['area_fraction'] = (video_data['area']/video_data['area'].quantile(0.99)) * 100

        fs = int(1/video_data['time'].diff().median())
        if fs != 25:
            print('skipping {} {}'.format(subj, ses))
            return pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
        
        measure = 'area_psc'
        epochs_p_stim = make_epochs(df=video_data, df_meta=df, locking='time_stim', 
                                    start=-2, dur=6, measure=measure, fs=fs, 
                                    baseline=False, b_start=-1, b_dur=1)
        epochs_p_lick = make_epochs(df=video_data, df_meta=df, locking='time_lick', 
                                    start=-2, dur=6, measure=measure, fs=fs, 
                                    baseline=False, b_start=-1, b_dur=1)

        # interpolated?
        df['pupil'] = 1
        df['video_int'] = [video_data.loc[(video_data['time']>start)&(video_data['time']>end), 'int'].mean() for start, end in zip(df['time_stim']-2, df['time_stim'])]
        df['video_good'] = [video_data.loc[(video_data['time']>start)&(video_data['time']>end), 'isgood'].mean() for start, end in zip(df['time_stim']-2, df['time_stim'])]

        # append:
        columns = ['subject_id', 'session_id', 'trial_id', 'trial_type', 'outcome']
        epochs_p_stim[columns] = df[columns]
        epochs_p_stim = epochs_p_stim.set_index(columns)
        epochs_p_lick[columns] = df[columns]
        epochs_p_lick = epochs_p_lick.set_index(columns)
        
    else:
        df['pupil'] = 0
        epochs_p_stim = pd.DataFrame([])
        epochs_p_lick = pd.DataFrame([])

    epochs_s_stim = []
    epochs_s_lick = []
    epochs_s_stim_z = []
    epochs_s_lick_z = []    
    
    if os.path.exists(os.path.join(raw_dir, exp, subj, ses, 'spikeData.mat')):
        df['spike'] = 1
        spike_mat = loadmat(os.path.join(raw_dir, exp, subj, ses, 'spikeData.mat'))['spikeData']
        data = [[row.flat[0] for row in line] for line in spike_mat[0][0]]
        df_spike_mat = pd.DataFrame(data).T
        df_spike_mat.columns = spike_mat.dtype.names

        df_spike_mat = df_spike_mat.applymap(lambda x: list(x) if (type(x)==np.ndarray) else x)
        df_spike_mat = df_spike_mat.applymap(lambda x: np.nan if x==[] else x)
        df_spike_mat = df_spike_mat.applymap(lambda x: x[0] if (type(x)==list) else x)
        df_spike_mat = df_spike_mat.applymap(lambda x: list(x) if (type(x)==np.ndarray) else x)
        df_spike_mat = df_spike_mat.applymap(lambda x: np.nan if x==[] else x)

        edges = np.arange(np.floor(df['time_trial_start'].min()-1), 
                            np.floor(df['time_trial_start'].max()+30), 
                            bin_width).round(3)

        max_ISI_FA     = 0.01  # Select on fraction of spikes within absolute refractory period (1.5ms)
        min_iso_dis    = 5     # Select on Isolation Distance
        min_fr_avg     = 0.2   # Select on minimum average firing rate throughout the session
        min_coverage   = 0.7   # Select on coverage throughout the session (fraction of session):
        sign_resp      = 0     # Whether any condition needs to show a significant response
        for i in range(df_spike_mat.shape[0]):
            
            cell_id = df_spike_mat['cell_ID'].iloc[i]
            cell_type = int(df_spike_mat['celltype'].iloc[i])
            area = df_spike_mat['area'].iloc[i]
            layer = df_spike_mat['Layer'].iloc[i]
            isi_fa = df_spike_mat['QM_ISI_FA'].iloc[i]
            iso_dis = df_spike_mat['QM_IsolationDistance'].iloc[i]
            fr_avg = df_spike_mat['avg_fr'].iloc[i]
            coverage = df_spike_mat['coverage'].iloc[i]
            if isi_fa > max_ISI_FA:
                print('skipping neuron {}'.format(cell_id))
                continue
            if iso_dis < min_iso_dis:
                print('skipping neuron {}'.format(cell_id))
                continue
            if fr_avg < min_fr_avg:
                print('skipping neuron {}'.format(cell_id))
                continue
            if coverage < min_coverage:
                print('skipping neuron {}'.format(cell_id))
                continue
            
            spike_times = np.array(df_spike_mat['ts'].iloc[i]).ravel() / micro_to_s # in seconds
            
            # make time series:
            df_spike = pd.DataFrame({'time': (edges + (bin_width/2))[:-1],
                                    'spike': np.histogram(spike_times, edges)[0]})
            
            # convolve:
            dx = df_spike['time'].diff().median().round(3)
            x = np.arange(-win_len/2, (win_len/2)+dx, dx)
            kernel = gauss(x, a=1, mu=0, sig=sigma)
            kernel = kernel[x>=0]
            df_spike['spike_gaus'] =  np.convolve(df_spike['spike'], kernel, mode="full")[:-len(kernel)+1]

            # plt.plot(df_spike['spike'])
            # plt.plot(df_spike['spike_gaus'])

            # make epochs:
            measure = 'spike_gaus'
            epoch_stim = make_epochs(df=df_spike, df_meta=df, locking='time_stim', 
                                        start=-0.5, dur=2, measure=measure, fs=int(1/dx), 
                                        baseline=False, b_start=-1, b_dur=1)
            epoch_lick = make_epochs(df=df_spike, df_meta=df, locking='time_lick', 
                                        start=-0.5, dur=2, measure=measure, fs=int(1/dx), 
                                        baseline=False, b_start=-1, b_dur=1)

            # append:
            columns = ['subject_id', 'session_id', 'trial_id', 'trial_type', 'outcome']
            epoch_stim[columns] = df[columns]
            epoch_stim[['cell_id', 'cell_type', 'area', 'layer', 'isi_fa', 'iso_dis', 'fr_avg', 'coverage']] = [cell_id, cell_type, area, layer, isi_fa, iso_dis, fr_avg, coverage]
            epoch_stim = epoch_stim.set_index(['subject_id', 'session_id', 'trial_id', 'trial_type', 'outcome', 
                                                        'cell_id', 'cell_type', 'area', 'layer', 'isi_fa', 'iso_dis', 'fr_avg', 'coverage'])
            
            
            epoch_lick[columns] = df[columns]
            epoch_lick[['cell_id', 'cell_type', 'area', 'layer', 'isi_fa', 'iso_dis', 'fr_avg', 'coverage']] = [cell_id, cell_type, area, layer, isi_fa, iso_dis, fr_avg, coverage]
            epoch_lick = epoch_lick.set_index(['subject_id', 'session_id', 'trial_id', 'trial_type', 'outcome', 
                                                        'cell_id', 'cell_type', 'area', 'layer', 'isi_fa', 'iso_dis', 'fr_avg', 'coverage'])
            
            # zscore:
            epoch_stim_z = epoch_stim.copy()
            epoch_lick_z = epoch_lick.copy()
            x = epoch_stim_z.columns
            ind = (epoch_stim.index.get_level_values('trial_type') != 'probe')
            baselines = np.array(epoch_stim.loc[ind, (x>-1)&(x<-0.2)].mean(axis=1)).ravel()[1:-1]
            epoch_stim_z = (epoch_stim_z - baselines.mean()) / baselines.std()
            epoch_lick_z = (epoch_lick_z - baselines.mean()) / baselines.std()

            # append:
            epochs_s_stim.append(epoch_stim)
            epochs_s_lick.append(epoch_lick)
            epochs_s_stim_z.append(epoch_stim_z)
            epochs_s_lick_z.append(epoch_lick_z)

        epochs_s_stim = pd.concat(epochs_s_stim)
        epochs_s_lick = pd.concat(epochs_s_lick)
        epochs_s_stim_z = pd.concat(epochs_s_stim_z)
        epochs_s_lick_z = pd.concat(epochs_s_lick_z)
    else:
        df['spike'] = 0
        epochs_s_stim = pd.DataFrame([])
        epochs_s_lick = pd.DataFrame([])
        epochs_s_stim_z = pd.DataFrame([])
        epochs_s_lick_z = pd.DataFrame([])

    return df, epochs_p_stim, epochs_p_lick, epochs_s_stim, epochs_s_lick, epochs_s_stim_z, epochs_s_lick_z

def remove_sessions(df):

    # counts:
    counts = df.groupby(['subject_id', 'session_id']).count()['trial_id']
    counts = counts[counts<50]
    for (subj, ses), c in counts.groupby(['subject_id', 'session_id']):
        print('removing subj {} ses {} (fewer than 50 trials...)'.format(subj, ses))
        exclude = np.array((df['subject_id']==subj) & (df['session_id']==ses))
        df = df.loc[~exclude,:].reset_index(drop=True)
    print()

    # # poor accuracy:
    # accuracy = df.groupby(['subject_id', 'session_id', 'trial_type'])['correct'].mean().groupby(['subject_id', 'session_id']).mean()
    # # accuracy = df.loc[df['difficulty']>=4,:].groupby(['subject_id', 'session_id', 'trial_type'])['correct'].mean().groupby(['subject_id', 'session_id']).mean()
    # for (subj, ses), a in accuracy.groupby(['subject_id', 'session_id']):
    #     print(float(a))
    #     if float(a) < 0.3:
    #         print('removing subj {} ses {} (overall accuracy less than 50%...)'.format(subj, ses))
    #         exclude = np.array((df['subject_id']==subj) & (df['session_id']==ses))
    #         df = df.loc[~exclude,:].reset_index(drop=True)
    # print()

    return df

def match_meta_epochs(df, epochs):

    columns = ['subject_id', 'session_id', 'trial_id', 'trial_type', 'outcome']
    
    # intersection:
    if len(epochs.index.names) == 13:
        drop = list(range(5,13))
        df = df[df.set_index(columns).index.isin(epochs.index.droplevel(drop))].reset_index(drop=True)
        epochs = epochs[epochs.index.droplevel(drop).isin(df.set_index(columns).index)]
    if len(epochs.index.names) == len(columns):
        df = df[df.set_index(columns).index.isin(epochs.index)].reset_index(drop=True)
        epochs = epochs[epochs.index.isin(df.set_index(columns).index)]

    return df, epochs

def select_data_behavior(df):

    df['subject_id'] = df['subject_id'].astype(str)
    df['session_id'] = df['session_id'].astype(str)

    # exclude:
    exclude = np.array( 
                        
                        # (df['experiment']!='MST') |
                        (df['rt']<0.2) | 
                        ((df['outcome']==0)&(df['rt'].isna())) |
                        (df['outcome_p'].isna()) |
                        (df['trial_type']=='probe') |
                        (df['trial_type']=='conflict') |
                        (df['trial_dur'] < 3) |
                        (df['trial_dur'] > 20) |
                        (df['trial_id'] < 1) |
                        # (df['trial_id']<25) |
                        # (df['trial_id']>350) |
                        (np.zeros(df.shape[0], dtype=bool))
                        )
    print()
    print('excluding {}% of data: experiment!=MST; trial==probe; trial_type==conflict:'.format(exclude.mean()*100))
    print()
    df = df.loc[~exclude,:].reset_index()

    # # remove last 20 trials:
    # dfs = []
    # for (subj, ses), d in df.groupby(['subject_id', 'session_id']):
    #     dfs.append(d.loc[d['trial_id'] < (d['trial_id'].max()-20),:])
    # df = pd.concat(dfs)

    # remove sessions:
    df = remove_sessions(df)

    # print
    print('left with {} mice, and {} sessions'.format(df.groupby(['subject_id']).count().shape[0], df.groupby(['subject_id', 'session_id']).count().shape[0]))

    return df.reset_index(drop=True)

def select_data_pupil(df, epochs_p_stim, epochs_p_lick):

    # match:
    df, epochs_p_stim = match_meta_epochs(df.loc[df['pupil']==1,:], epochs_p_stim)
    df, epochs_p_lick = match_meta_epochs(df.loc[df['pupil']==1,:], epochs_p_lick)

    # add variables:
    x = epochs_p_stim.columns
    df['pupil_stim_b'] = epochs_p_stim.loc[:, (x>-1)&(x<0)].mean(axis=1).values
    x = epochs_p_lick.columns
    df['pupil_lick'] = epochs_p_lick.loc[:, (x>0.5)&(x<3.5)].mean(axis=1).values - epochs_p_lick.loc[:, (x>-1)&(x<0)].mean(axis=1).values

    # df['pupil_lick'] = epochs_p_lick.loc[:, (x>0.4)&(x<1.2)].diff(axis=1).mean(axis=1).values


    # exclude:
    exclude = np.array( 
                        (df['video_int']>0.01) |
                        (df['pupil_stim_b'].isna())
                        )

    print()
    print('excluding {}% of data'.format(exclude.mean()*100))
    print()
    df = df.loc[~exclude,:].reset_index()

    # remove sessions:
    df = remove_sessions(df)

    # match:
    df, epochs_p_stim = match_meta_epochs(df, epochs_p_stim)
    df, epochs_p_lick = match_meta_epochs(df, epochs_p_lick)

    # print
    print('left with {} mice, and {} sessions'.format(df.groupby(['subject_id']).count().shape[0], df.groupby(['subject_id', 'session_id']).count().shape[0]))

    # # pupil quality:
    # pupil = df.groupby(['subject_id', 'session_id'])['pupil_lick'].mean()
    # for (subj, ses), p in pupil.groupby(['subject_id', 'session_id']):
    #     if bool(p.values<6):
    #         print('removing subj {} ses {}'.format(subj, ses))
    #         exclude = np.array((df['subject_id']==subj) & (df['session_id']==ses))
    #         df = df.loc[~exclude,:].reset_index(drop=True)
    #         epochs_p_stim = epochs_p_stim.loc[~exclude,:]
    #         epochs_p_lick = epochs_p_lick.loc[~exclude,:]

    return df.reset_index(drop=True), epochs_p_stim, epochs_p_lick

def select_data_spike(df, epochs_s_stim, epochs_s_lick):

    df = df.loc[df['spike']==1,:]
    # df = df.loc[df['subject_id']!=2030,:]

    # remove sessions:
    df = remove_sessions(df)

    # # intersection:
    # df['subject_id'] = df['subject_id'].astype(str)
    # df['session_id'] = df['session_id'].astype(str)
    # df = df[df.set_index(columns).index.isin(epochs_s_stim.index.droplevel(drop))].reset_index(drop=True)
    # df = df[df.set_index(columns).index.isin(epochs_s_lick.index.droplevel(drop))].reset_index(drop=True)
    # epochs_s_stim = epochs_s_stim[epochs_s_stim.index.droplevel(drop).isin(df.set_index(columns).index)]
    # epochs_s_lick = epochs_s_lick[epochs_s_lick.index.droplevel(drop).isin(df.set_index(columns).index)]
    # means = epochs_s_stim.groupby(['session_id', 'outcome']).mean().groupby(['session_id']).mean()
    # x = means.columns
    # early = means.loc[:,(x>0)&(x<0.25)].mean(axis=1).reset_index()
    # early = early.loc[(~np.isnan(early[0]))&(early[0]<1)]
    # df = df.loc[df['session_id'].isin(early['session_id'])]

    # match:
    # df, epochs_s_stim = match_meta_epochs(df, epochs_s_stim.query("(area=='V1')"))
    # df, epochs_s_lick = match_meta_epochs(df, epochs_s_lick.query("(area=='V1')"))

    df, epochs_s_stim = match_meta_epochs(df, epochs_s_stim)
    df, epochs_s_lick = match_meta_epochs(df, epochs_s_lick)

    # print
    print('left with {} mice, and {} sessions'.format(df.groupby(['subject_id']).count().shape[0], df.groupby(['subject_id', 'session_id']).count().shape[0]))

    return df.reset_index(drop=True), epochs_s_stim, epochs_s_lick

def plot_pupil_responses(epochs_p_stim, epochs_p_lick):

    fig = plt.figure(figsize=(4,2))

    ax = fig.add_subplot(121)
    means = epochs_p_stim.groupby(['subject_id', 'outcome']).mean().groupby(['outcome']).mean()
    sems = epochs_p_stim.groupby(['subject_id', 'outcome']).mean().groupby(['outcome']).sem()
    x = means.columns
    for i, l in zip([0,1], ['H', 'M']):
        plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], alpha=0.1)
        plt.plot(x, means.iloc[i], label=l)
    plt.axvspan(-0.5, 0, color='grey', alpha=0.1)
    plt.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.ylim(45,70)
    plt.xlabel('Time from stimulus change (s)')
    plt.ylabel('Pupil size (% max)')

    ax = fig.add_subplot(122)
    means = epochs_p_lick.groupby(['subject_id', 'outcome']).mean().groupby(['outcome']).mean()
    sems = epochs_p_lick.groupby(['subject_id', 'outcome']).mean().groupby(['outcome']).sem()
    x = means.columns
    for i, l in zip([0], ['H']):
        plt.fill_between(x, means.iloc[i]-sems.iloc[i], means.iloc[i]+sems.iloc[i], alpha=0.1)
        plt.plot(x, means.iloc[i], label=l)
    plt.axvline(0, color='black', lw=0.5)
    plt.legend()
    plt.ylim(45,70)
    plt.xlabel('Time from correct lick (s)')
    plt.ylabel('Pupil size (% max)')

    sns.despine(trim=False)
    plt.tight_layout()
    return fig

def pupil_behavior(df, groupby='pupil_b_bin', x='pupil_b'):

    colormap = 'dark:salmon'

    fig = plt.figure(figsize=(6,4))
    plt_nr = 1
    for diff_split in [0,1]:
        for i, measure, label in zip(range(1,4), 
                                    ['fa_rate', 'correct', 'rt'],
                                    ['FA rate (bouts / s)', 'Accuracy (% correct)', 'RT (s)']):
            ax = fig.add_subplot(2,3,plt_nr)
            if diff_split == 0:
                means = df.groupby(['subject_id', groupby]).mean().groupby([groupby]).mean().reset_index()
                sems = df.groupby(['subject_id', groupby]).mean().groupby([groupby]).sem().reset_index()
                plt.errorbar(x=means[x], 
                                y=means[measure],
                                yerr=sems[measure],
                                fmt='o',
                                markerfacecolor='lightgrey',
                                markeredgecolor=sns.color_palette(colormap,5)[0],
                                ecolor=sns.color_palette(colormap,5)[0],
                                elinewidth=1)
                poly = np.polyfit(x=means[x], 
                                y=means[measure], deg=2)
                plt.plot(means[x], 
                            np.polyval(poly, means[x]),
                            color=sns.color_palette(colormap,5)[0], lw=1)
            else:
                means = df.groupby(['subject_id', 'difficulty', groupby]).mean().groupby(['difficulty', groupby]).mean().reset_index()
                sems = df.groupby(['subject_id', 'difficulty', groupby]).mean().groupby(['difficulty', groupby]).sem().reset_index()        
                for d in [2,3,4,5]:
                    plt.errorbar(x=means.loc[means['difficulty']==d, x], 
                                    y=means.loc[means['difficulty']==d, measure],
                                    yerr=sems.loc[sems['difficulty']==d, measure],
                                    fmt='o',
                                    markerfacecolor='lightgrey',
                                    markeredgecolor=sns.color_palette(colormap,5)[d-1],
                                    ecolor=sns.color_palette(colormap,5)[d-1],
                                    elinewidth=1)
                    poly = np.polyfit(x=means.loc[means['difficulty']==d, x], 
                                    y=means.loc[means['difficulty']==d, measure], deg=2)
                    plt.plot(means.loc[means['difficulty']==d, x], 
                                np.polyval(poly, means.loc[means['difficulty']==d, x]),
                                color=sns.color_palette(colormap,5)[d-1], lw=1)
            plt.xlabel('Pupil size (% max)')
            plt.ylabel(label)
            plt_nr += 1
    sns.despine(trim=False)
    plt.tight_layout()
    return fig

def bias_modality_analysis(df, groupby=['subject_id', 'session_id']):
    
    correct_visual = df.loc[df['trial_type']=='visual'].groupby(groupby)['correct'].mean()
    correct_auditory = df.loc[df['trial_type']=='auditory'].groupby(groupby)['correct'].mean()
    modality_bias = correct_visual - correct_auditory
    performance = (correct_visual + correct_auditory) / 2

    res = pd.concat((performance, modality_bias), axis=1)
    res.columns = ['performance', 'bias']


    # fig = plt.figure(figsize=(2,2))
    # ax = fig.add_subplot(111)
    # plt.scatter(performance, modality_bias, alpha=0.25)
    # poly = np.polyfit(x=correct_visual, 
    #                 y=correct_auditory, deg=1)
    # plt.plot(correct_visual, 
    #             np.polyval(poly, correct_visual),
    #             color='k', lw=1)
    # r,p = sp.stats.pearsonr(correct_visual, correct_auditory)
    # plt.title('r = {}, p = {}'.format(round(r,3), round(p,3)))

    fig = sns.jointplot(data=res, x="performance", y="bias", height=2.5)
    plt.axhline(0, lw=0.5, color='k')
    plt.xlabel('Performance\n(Vis. + vs. Aud.)/2')
    plt.ylabel('Bias\n(Vis. vs. Aud.)')
    
    # sns.despine(trim=False)
    # plt.tight_layout()
    
    return fig

def bias_modality_pupil_analysis(df, groupby=['subject_id', 'session_id']):
    
    correct_visual = df.loc[df['trial_type']=='visual'].groupby(groupby)['correct'].mean()
    correct_auditory = df.loc[df['trial_type']=='auditory'].groupby(groupby)['correct'].mean()
    modality_bias = correct_visual - correct_auditory
    pupil_correct_visual = df.loc[(df['trial_type']=='visual')&(df['correct']==1),:].groupby(groupby)['pupil_stim_b'].mean()
    pupil_correct_auditory = df.loc[(df['trial_type']=='auditory')&(df['correct']==1),:].groupby(groupby)['pupil_stim_b'].mean()
    pupil_modality_bias = pupil_correct_visual - pupil_correct_auditory
    res = pd.concat((modality_bias, pupil_modality_bias), axis=1)

    fig = plt.figure(figsize=(2,2))
    
    ax = fig.add_subplot(111)
    
    # for (subj, ses), d in res.groupby(groupby):
    #     plt.scatter(d['correct'], d['pupil_stim_b'])
    plt.scatter(res['correct'], res['pupil_stim_b'])
    poly = np.polyfit(x=res['correct'], 
                    y=res['pupil_stim_b'], deg=1)
    plt.plot(res['correct'], 
                np.polyval(poly, res['correct']),
                color='k', lw=1)
    plt.axhline(0, lw=0.5, color='k')
    plt.axvline(0, lw=0.5, color='k')
    r,p = sp.stats.pearsonr(res['correct'], res['pupil_stim_b'])
    plt.title('r = {}, p = {}'.format(round(r,3), round(p,3)))
    plt.xlabel('Modality bias\n(Vis. vs Aud. accuracy)')
    plt.ylabel('Pupil modality bias\n(pupil size on correct Vis. vs\ncorrect Aud. trials)')

    sns.despine(trim=False)
    plt.tight_layout()
    
    return fig
