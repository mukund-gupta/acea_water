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
df = pd.read_csv(files[0])

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


def preprocess_int(df, col_ind, start_ind=3955):
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
col_ind_target = 13
col_targets = col_inds[1]
col_ind_others = [1, 15, 16]
col_rain = col_inds[0]
col_vol = col_inds[3]
# Get target time series data
target_ts, target_name = preprocess_int(df, col_ind_target)
target_length = target_ts.size
# Get time series data for other variables
other_ts = []
other_ts_int = []
other_name = []
for n in range(len(col_ind_others)):
    ts_, name_ = preprocess(df, col_ind_others[n])
    ts_int_, name_ = preprocess_int(df, col_ind_others[n])
    other_ts.append(ts_)
    other_ts_int.append(ts_int_)
    other_name.append(name_)
# Get time series data for all targets
all_target_ts = []
all_target_name = []
for n in range(len(col_targets)):
    ts_, name_ = preprocess_int(df, col_targets[n])
    all_target_ts.append(ts_)
    all_target_name.append(name_)
# Get time series data for all rain variables
rain_ts = []
rain_name = []
for n in range(len(col_rain)):
    ts_, name_ = preprocess_int(df, col_rain[n])
    rain_ts.append(ts_)
    rain_name.append(name_)
# Get time series data for all volume variables
vol_ts = []
vol_name = []
for n in range(len(col_vol)):
    ts_, name_ = preprocess_int(df, col_vol[n])
    vol_ts.append(ts_)
    vol_name.append(name_)
    
def plot_data_preprocessed(df, col_ind):
    data_ts, _ = preprocess(df, col_ind)
    data_ts_int, _ = preprocess_int(df, col_ind)
    data_ts_size = data_ts.size
    data_ts_int_size = data_ts_int.size
    time_array = np.linspace(0, data_ts_size-1, data_ts_size)
    time_array2 = np.linspace(0, data_ts_int_size-1, data_ts_int_size)
    plt.figure()
    plt.scatter(time_array, data_ts, s=0.2, alpha=0.6)
    plt.scatter(time_array2, data_ts_int, s=0.2, alpha=0.6)
    plt.show()
    
# plot_data_preprocessed(df, col_ind_target) 
    
"""
# Calculate spearmans rank correlation for different lags
# Note that the target data is appended to the list of other variables so
# it is included in the analysis (effectively an autocorrelation)
all_ts = [target_ts] + other_ts
all_names = [target_name] + other_name
num_datasets = len(all_ts)
# loop through all datasets to calculate n-lag spearmans rank coefficients
ccl = np.empty((num_datasets, target_length-1))
for n in range(num_datasets):
    ccl[n, :] = cross_corr_lag(target_ts, all_ts[n])
lags = np.linspace(0, ccl.shape[1]-1, ccl.shape[1])

# Plot results
plt.figure()
# fig, axes = plt.subplots(num_datasets, 1, sharex=True, sharey=True)
for n in range(num_datasets):
    plt.plot(ccl[n], label=all_names[n], lw=1, alpha=0.7)
    # axes[n].set_ylim([-1, 1])
plt.legend()
plot_title = 'Correlation lag-N: ' + target_name
plt.title(plot_title)
plt.xlabel('')
plt.show()
    """
# plt.figure()
# plt.plot(target_ts, label=target_name, lw=1, alpha=0.7)
# plt.title('Time series')
# plt.show()
winlength = target_length
tau = 20
t = np.linspace(0, winlength-1, winlength)
exp_window = np.exp(-t/tau)
exp_model = np.convolve(other_ts[0], exp_window, 'full')
exp_model_av = np.convolve(moving_average(other_ts[0], 7),
                        exp_window, 'full')

"""
plt.figure()
for n in range(len(col_targets)):
    plt.plot(normalise_0_to_1(all_target_ts[n]), label=all_target_name[n], lw=1, alpha=0.7)
# plt.plot(other_ts[0], label=other_name[0], lw=1, alpha=0.7)
# plt.plot(moving_average(other_ts[0], 30), label='Mov av, win=30', lw=1, alpha=0.7)
plt.plot(normalise_0_to_1(exp_model), label='exp model', lw=1, alpha=0.7)
# plt.plot(normalise_0_to_1(exp_model_av), label='exp model av', lw=1, alpha=0.7)
# plt.plot(moving_average(other_ts[0], 90), label='Mov av, win=90', lw=1, alpha=0.7)
plt.legend()
plt.title('Time series')
plt.show()


plt.figure()
for n in range(len(col_rain)):
    plt.plot(rain_ts[n], label=rain_name[n], lw=1, alpha=0.7)
plt.legend()
plt.title('Time series')
plt.show()


plt.figure()
for n in range(len(col_vol)):
    plt.plot(normalise_0_to_1(vol_ts[n]), label=vol_name[n], lw=1, alpha=0.7)
plt.plot(normalise_0_to_1(all_target_ts[0]), label=all_target_name[0], lw=1, alpha=0.7)
plt.legend()
plt.title('Time series')
plt.show()
"""


# plt.figure()
# vol_total = normalise_0_to_1(np.sum(vol_ts, 0))
# vol_total_deviation = vol_total - np.convolve(vol_total, np.ones(500), 'same') / 500
# vol_dev_smooth = np.convolve(vol_total_deviation, np.ones(10), 'same') / 10
# target_total = np.sum(all_target_ts, 0)
# # plt.plot(vol_total, label='total vol', lw=1, alpha=0.7)
# # plt.plot(vol_total_deviation, label='vol deviation', lw=1, alpha=0.7)
# plt.plot(normalise_0_to_1(vol_dev_smooth), label='vol dev smooth', lw=1, alpha=0.7)
# plt.plot(normalise_0_to_1(exp_model), label='exp model', lw=1, alpha=0.7)
# plt.plot(normalise_0_to_1(target_total[:-40]), label='total groundwater', lw=1, alpha=0.7)
# plt.legend()
# plt.title('Time series')
# plt.show()
# # cc = ccf(target_ts, other_ts)
# # cc_lag = ccf(target_ts, other_ts_lag)

# plt.figure()
# plt.plot(target_ts, label=target_name)
# plt.plot(other_ts, label=other_name)
# plt.plot(other_ts_lag, label='lag')
# plt.plot(cc, label='cross-correlation')
# plt.plot(cc_lag, label='cross-correlation_lag')
# plt.legend()
# plt.show()


"""
Add time lag?
Smoothing for deeper groundwater?
"""

def find_tau_correlation(rain_ts, target_ts, tau_array=None):
    """Calculate convolution of rainfall data with an exponential window
    with time constant tau, for a range of values of tau.
    Then determine how correlated these convolved signals are to the
    target data by calculating Spearman's rank correlation coefficient
    for each value of tau"""
    if tau_array is None:
        tau_array = np.linspace(1, 100, 100)
    rain_ts = np.asarray(rain_ts)
    target_ts = np.asarray(target_ts)
    winlength = rain_ts.size
    target_len = target_ts.size
    t = np.linspace(0, winlength-1, winlength)
    src = np.empty(len(tau_array))
    for n in range(len(tau_array)):
        exp_win = np.exp(-t/tau_array[n])
        rain_conv = np.convolve(rain_ts, exp_win, 'full')[:target_len]
        rain_conv = rain_conv / np.sum(exp_win)
        # if n==40:
        #     plt.figure()
        #     plt.plot(rain_ts, label='rain_ts')
        #     plt.plot(rain_conv, label='rain_conv')
        #     plt.plot(target_ts, label='target_ts')
        #     plt.legend()
        #     plt.show()
        src[n], _ = stats.spearmanr(target_ts, rain_conv)
    return src, tau_array



# # Calculate and plot correlation of convolved rainfall signal with the
# # target signal for different time constants of exponential window
# target_ind = 4
# plt.figure()
# for n in range(len(rain_name)):
#     src, tau_array = find_tau_correlation(
#                             normalise_0_to_1(rain_ts[n]),
#                             normalise_0_to_1(all_target_ts[target_ind]),
#                             tau_array=None)
#     plt.plot(tau_array, src, label=rain_name[n])   
# plt.xlabel("tau for exponential window")
# plt.ylabel("Spearman's Rank Coefficient")
# title_text = 'Correlation with target ' + all_target_name[target_ind] + \
#              ' for different rainfall data'
# plt.title(title_text)
# plt.legend()
# plt.show()


# LSTM
import numpy
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1, chunk_step=1, predict_steps=1):
    numchunk = int(np.floor((dataset.shape[1] - look_back - 1) / chunk_step))
    dataX = np.empty((numchunk, look_back, dataset.shape[0]))
    dataY = np.empty((numchunk, predict_steps))
    y_ind = []
    # Create chunks of data with the specified look back
    for i in range(numchunk):
        start_ind = chunk_step*i
        dataX[i, :, :] = dataset[:, start_ind:(start_ind + look_back)].T
        dataY[i, :] = dataset[0,start_ind+look_back:start_ind+look_back+predict_steps] #MG
        #dataY.append(dataset[0, start_ind + look_back])
        y_ind.append(start_ind + look_back)
    # Randomise order of chunks
    rand_indices = np.random.permutation(numchunk)
    x = numpy.array(dataX)
    y = numpy.array(dataY)
    y_ind = numpy.array(y_ind)
    x = x[rand_indices, :]
    y = y[rand_indices,:]
    #y = np.reshape(y, (y.size, 1)) # MG
    y_ind = y_ind[rand_indices]
    return x, y, y_ind

# Rain preprocessing
tau = 15
exp_win = np.exp(-t/tau)
exp_win = exp_win / np.sum(exp_win)
rain_len = rain_ts[0].size
rain_conv = []
for n in range(len(rain_ts)):
    conv = np.convolve(rain_ts[n], exp_win, 'full')[:rain_len]
    rain_conv.append(conv)

# Model parameters
target_ind = 1
look_back = 30
chunk_step = 30
train_ratio = 0.67
num_epochs = 100
batch_size = 5
predict_steps = 1

# fix random seed for reproducibility
numpy.random.seed(7)

# Remove zeros from target
# N.B. SHOULD REPLACE THIS WITH A DIFFERENT METHOD
for n in range(len(all_target_ts[target_ind])):
    if n > 0 and all_target_ts[target_ind][n] == 0:
        all_target_ts[target_ind][n] = all_target_ts[target_ind][n-1]
        
# normalize the datasets
target_data = all_target_ts[target_ind]
target_data = np.reshape(target_data, (target_data.size, 1))
scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = scaler.fit_transform(target_data)
target_scaled = np.squeeze(target_scaled)
rain_scaled = normalise_0_to_1(rain_conv[0])

# Combine the dataset into single array
dataset = np.stack((target_scaled, rain_scaled))

# Plot presprocessed data
plt.figure()
plt.plot(dataset[0, :], label=all_target_name[target_ind])
plt.plot(dataset[1, :], label=rain_name[0])
plt.legend()
plt.title('Preprocessed data')
plt.show()

#%% Data pre-processing for LSTM

# Split data into chunks with random order
x, y, y_ind = create_dataset(dataset, look_back=look_back,
                             chunk_step=chunk_step, predict_steps=predict_steps)
numchunk = y.shape[0]

# split into train and test sets
train_size = int(numchunk * train_ratio)
test_size = numchunk - train_size
trainX, testX = x[0:train_size, :, :], x[train_size:numchunk, :, :]
trainY, testY = y[0:train_size, :], y[train_size:numchunk, :]
trainYind, testYind = y_ind[0:train_size], y_ind[train_size:numchunk]

## Plot one example of chunk
#sample_num = 10
#plt.figure()
#time_ = np.arange(look_back)
#plt.plot(time_, trainX[sample_num, :, 0], label='Input: target')
#plt.plot(time_, trainX[sample_num, :, 1], label='Input: rain')
#plt.plot([look_back], trainY[sample_num, 0], 'xr', label='Output: target')
#plt.legend()
#plt.title("Example of one chunk from dataset")
#plt.show()

#%% Create and fit the LSTM network
num_features = x.shape[2]
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, num_features)))
model.add(Dense(predict_steps)) # MG
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=1)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)


## calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[:, 0], trainPredict[:,0]))
#print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[:, 0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))

#%% Plotting sample

# Plot one prediction example from test sample
pred_sample = 20
pred_ind = testYind[pred_sample]
plt.figure()
sample_t = np.linspace(pred_ind - look_back - 1, pred_ind - 1, look_back)
sample_t = sample_t.astype(np.int)
plt.plot(sample_t, target_data[sample_t[0]:sample_t[-1]])
plt.plot(np.arange(pred_ind,pred_ind+predict_steps), testPredict[pred_sample,:])
plt.plot(np.arange(pred_ind,pred_ind+predict_steps), target_data[sample_t[-1]:sample_t[-1]+predict_steps])
plt.title("Example of one prediction from test data")
plt.show()

#%% Plot all predictions from test samples against original data

plt.figure()
plt.plot(np.linspace(0, target_data.size-1, target_data.size), target_data,
         label='Original data')
plt.plot(testYind, testPredict[:, 0], 'xr', label='Test predictions')
plt.plot(trainYind, trainPredict[:, 0], 'xg', label='Train predictions')
plt.legend()
plt.title("Predictions compared to original dataset")
plt.show()

#%% Plot all predictions from test samples against original data

plt.figure()
plt.plot(np.linspace(0, target_data.size-1, target_data.size), target_data,
         label='Original data')
for ii in range(len(testYind)):
    plt.plot(np.arange(testYind[ii],testYind[ii] + predict_steps), testPredict[ii, :],color='k')
plt.legend()
plt.title("Predictions compared to original dataset")
plt.show()


