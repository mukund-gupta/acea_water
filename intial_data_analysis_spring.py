
"""
ACEA WATER

Initial Data Analysis

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from statsmodels.tsa.stattools import ccf
from scipy import stats


# Get all csv files from data
foldpath = r"acea-water-prediction"
files = list(Path(foldpath).rglob('*.csv'))

# Create dataframe
df = pd.read_csv(files[-3])

def preprocess(df, col_ind, start_ind=0):
    """Some basic preprocessing of data from selected column of dataframe.
    This will probably need to be thought about more to deal with NaNs
    more effectively"""
    pd_series = df.iloc[start_ind:, col_ind]
    pd_values = pd_series.to_numpy()
    # max_value = np.max(np.abs(pd_values))
    # pd_values = pd_values / max_value
    name = df.columns[col_ind]
    return pd_values, name


def preprocess_int(df, col_ind, start_ind=0):
    """Some basic preprocessing of data from selected column of dataframe.
    This will probably need to be thought about more to deal with NaNs
    more effectively"""
    pd_series = df.iloc[start_ind:, col_ind]
    pd_series = pd_series.interpolate(method='linear')
    pd_series = pd_series.fillna(0)
    pd_values = pd_series.to_numpy()
    # max_value = np.max(np.abs(pd_values))
    # pd_values = pd_values / max_value
    name = df.columns[col_ind]
    return pd_values, name


def spearman_lag(data1, data2, lag):
    """Calculate Spearman's rank correlation coefficient between 2 datasets,
    with a lag applied to data2"""
    data_length = data1.size
    if lag > 0:
        data2_lag = np.zeros(data_length)
        data2_lag[lag:] = data2[:-lag] 
    else:
        data2_lag = data2
    src, _ = stats.spearmanr(data1, data2_lag)
    return src


def cross_corr_lag(data1, data2):
    """Calculate Spearman's rank correlation coefficient between 2 datasets,
    for a range of different lags applied to data2"""
    data_length = data1.size
    crosscorr_lag = np.empty(data_length - 1)
    for n in range(data_length - 1):
        crosscorr_lag[n] = spearman_lag(data1, data2, lag=n)
    return crosscorr_lag


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def normalise_0_to_1(signal):
    sig_min = np.min(signal)
    sig_max = np.max(signal)
    sig_norm = (signal - sig_min) / (sig_max - sig_min)
    return sig_norm


def normalise_sum(signal):
    """
    Normalises a signal so that the sum of all samples is unity.

    :param signal: An array of values representing the signal to be normalised.
    :returns: An array of values representing the normalised signal.

    """
    signal = np.asarray(signal, dtype=float)
    sum_ = np.sum(signal) if signal.any() else 1.

    return signal / sum_


def movingav(signal, winwidth, winfunc=None):
    """
    Calculates the moving average of a signal using chosen window function.

    :param signal: An array of values representing the signal to be smoothed.
    :param winwidth: The width of the smoothing window (number of samples).
    :param winfunc: The window function to use for smoothing. By default, a
        rectangular window will be used.
    :returns: An array of values representing the smoothed signal.
    :raises ValueError: If window width is negative.
    :raises ValueError: If window width exceeds length of input array.

    """
    numsamples = len(signal)

    hww = int(winwidth / 2.)
    winwidth = 2 * hww + 1

    if winwidth < 0:
        raise ValueError("window width must not be negative")

    if winwidth >= numsamples:
        raise ValueError("window width must not exceed length of input array")

    win = np.ones(winwidth) if winfunc is None else winfunc(winwidth)

    win = normalise_sum(win)

    halfwin2 = normalise_sum(win[hww:])
    halfwin1 = normalise_sum(win[:hww+1])

    valstart = np.dot(halfwin2, signal[0:hww+1])
    valend = np.dot(halfwin1, signal[numsamples - hww - 1:])

    signal = np.concatenate((np.ones(hww) * valstart, signal,
                             np.ones(hww) * valend))

    wpos = hww
    wend = len(signal) - hww - 1

    sig_smooth = np.empty(numsamples)

    while wpos <= wend:
        sig_smooth[wpos - hww] = np.dot(signal[wpos-hww: wpos+hww+1], win)
        wpos += 1

    return sig_smooth

    
def plot_data_preprocessed(df, col_ind):
    data_ts, data_name = preprocess(df, col_ind)
    data_ts_int, _ = preprocess_int(df, col_ind)
    data_ts_size = data_ts.size
    time_array = np.linspace(0, data_ts_size-1, data_ts_size)
    plt.figure()
    plt.plot(data_ts_int, alpha=0.8, zorder=1)
    plt.scatter(time_array, data_ts, s=5, color=[1, 0.8, 0.2, 1], zorder=2)
    plt.title(data_name)
    plt.show()


def find_datatypes(df):
    names = df.columns
    datatypes = ['Rainfall',
                 'Depth_to_Groundwater',
                 'Temperature',
                 'Volume',
                 'Hydrometry',
                 'Flow_rate',
                 'Lake_level']
    col_inds = []
    for n in range(len(datatypes)):
        col_ind_type = []
        for c in range(len(names)):
            if datatypes[n] in names[c]:
                col_ind_type.append(c)
        col_inds.append(col_ind_type)
    return datatypes, col_inds

datatypes, col_inds = find_datatypes(df)

# Get time series data for target variable and some other variables
# Specified by column index of the pandas dataframe
col_ind_target = 8
col_targets = [11, 12, 13, 14, 15]
col_ind_others = [1, 15, 16]
col_rain = np.linspace(1, 10, 10).astype(np.int)
col_vol = np.linspace(20, 24, 5).astype(np.int)
    
# plot_data_preprocessed(df, col_ind_target) 
    

"""Groundwater variables"""
col_gw = np.linspace(6, 8, 3).astype(np.int)
gw_ts = []
gw_name = []
for n in range(len(col_gw)):
    ts_, name_ = preprocess_int(df, col_gw[n], start_ind=3000)
    gw_ts.append(ts_)
    gw_name.append(name_)

plt.figure()
for n in range(len(col_gw)):
    plt.plot(-normalise_0_to_1(gw_ts[n]), label=gw_name[n])
    # plt.plot(gw_ts[n] - gw_ts[n][-1], label=gw_name[0], alpha=0.7)
plt.legend()
plt.show()


"""Temperature variables"""
col_temp = np.linspace(9, 11, 3).astype(np.int)
temp_ts = []
temp_name = []
for n in range(len(col_temp)):
    ts_, name_ = preprocess_int(df, col_temp[n], start_ind=0)
    temp_ts.append(ts_)
    temp_name.append(name_)

plt.figure()
for n in range(len(col_temp)):
    plt.plot(normalise_0_to_1(temp_ts[n]), label=temp_name[n])
    # plt.plottemp_ts[n] - temp_ts[n][-1], label=temp_name[0], alpha=0.7)
plt.legend()
plt.show()


"""Rainfall variables"""
col_rain = np.linspace(1, 5, 5).astype(np.int)
rain_ts = []
rain_name = []
for n in range(len(col_rain)):
    ts_, name_ = preprocess_int(df, col_rain[n], start_ind=5400)
    # ts_, name_ = preprocess(df, col_rain[n], start_ind=5400)
    rain_ts.append(ts_)
    rain_name.append(name_)

smoothing_win = 30
plt.figure()
for n in range(len(col_rain)):
    smoothed = moving_average(rain_ts[n], smoothing_win)
    plt.plot(smoothed, label=rain_name[n], alpha=0.7)
plt.legend()
plt.title(('Rainfall, smoothing window: ', smoothing_win))


"""Flow rate variables"""
col_flow = np.linspace(12, 15, 4).astype(np.int)
flow_ts = []
flow_ts_filt = []
flow_name = []
flow_mov_av = []
flow_thresh_hi = []
flow_thresh_lo = []
moving_av_win = 450
for n in range(len(col_flow)):
    ts_, name_ = preprocess_int(df, col_flow[n], start_ind=5400)
    # ts_, name_ = preprocess(df, col_flow[n], start_ind=5400)
    # mov_av = np.convolve(ts_, np.ones(moving_av_win), 'full') / moving_av_win
    # mov_av = mov_av[moving_av_win-1:]
    removed_zeros = np.where(ts_==0, np.nan, ts_)
    overall_av = np.nanmean(removed_zeros)
    filled_zeros = np.where(np.isnan(removed_zeros), overall_av, removed_zeros)
    mov_av = movingav(filled_zeros, moving_av_win, winfunc=None)
    thresh_hi = mov_av + 1.6*np.nanstd(removed_zeros)
    thresh_lo = mov_av - 1.6*np.nanstd(removed_zeros)
    flow_thresh_hi.append(thresh_hi)
    flow_thresh_lo.append(thresh_lo)
    is_outlier = np.logical_and(ts_ < thresh_hi, ts_ > thresh_lo)
    ts_filt = np.where(is_outlier, ts_, np.nan)
    flow_mov_av.append(mov_av)
    flow_ts.append(ts_)
    flow_ts_filt.append(ts_filt)
    flow_name.append(name_)

smoothing_win = 1
plt.figure()
for n in range(len(col_flow)):
    # plt.plot(normalise_0_to_1(flow_ts[n]), label=flow_name[n])
    # plt.plot(flow_ts[n] - flow_ts[n][-1], label=flow_name[n], alpha=0.7)
    smoothed = moving_average(flow_ts[n], smoothing_win)
    plt.plot(flow_mov_av[n], lw=1, color=[0, 0, 0, 0.5])
    plt.plot(flow_thresh_hi[n], lw=1, color=[0, 0, 0, 0.5])
    plt.plot(flow_thresh_lo[n], lw=1, color=[0, 0, 0, 0.5])
    plt.plot(smoothed, lw=1, color=[0, 0, 0, 0.7])
    plt.plot(flow_ts_filt[n], label=flow_name[n], alpha=0.9)
    
plt.legend()
plt.title('Flow rates')

plt.show()
plt.figure()
for n in range(len(col_flow)):
    smoothed = moving_average(flow_ts[n], smoothing_win)
    plt.plot(normalise_0_to_1(smoothed), label=flow_name[n], alpha=0.7)
plt.legend()
plt.title('Flow rates normalised')
plt.show()

