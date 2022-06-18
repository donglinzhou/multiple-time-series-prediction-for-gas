import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
from sklearn.preprocessing import StandardScaler

# 评估函数
def mape(y_true, y_pred):
    return (K.abs(y_true - y_pred) / K.abs(y_pred)) * 100

def smape(y_true, y_pred):
    return (K.abs(y_pred - y_true) / ((K.abs(y_true) + K.abs(y_pred))))*100

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# 切分数据集
def df_to_cnn_rnn_format(df, train_size=0.5, look_back=5, target_column='target', scale_X=True):
    """
    TODO: output train and test datetime

    Input is a Pandas DataFrame. 
    Output is a np array in the format of (samples, timesteps, features).
    Currently this function only accepts one target variable.

    Usage example:

    # variables
    df = data           # should be a pandas dataframe
    test_size = 0.5     # percentage to use for training
    target_column = 'c' # target column name, all other columns are taken as features
    scale_X = False
    look_back = 5       # Amount of previous X values to look at when predicting the current y value
    LSTM.py:data(6352,40),train_size=0.7,look_back=5*24,target_column='gasPower', saclx_X = True
    """
    df = df.copy()

    # Make sure the target column is the last column in the dataframe
    df['target'] = df[target_column]        # Make a copy of the target column
    df = df.drop(columns=[target_column])   # Drop the original target column
    
    target_location = df.shape[1] - 1           # column index number of target-last column is target column
    split_index = int(df.shape[0]*train_size)   # the index at which to split df into train and test

    # 划分训练集和测试集
    # ...train
    X_train = df.values[:split_index, :target_location]
    y_train = df.values[:split_index, target_location]

    # ...test
    X_test = df.values[split_index:, :target_location]      # original is split_index:-1
    y_test = df.values[split_index:, target_location]       # original is split_index:-1

    # Scale the features
    if scale_X:
        scalerX = StandardScaler()
        X_train = scalerX.fit_transform(X_train)
        X_test = scalerX.transform(X_test)

    # 提取需要预测的train和test
    # Reshape the arrays
    num_features = target_location  # All columns before the target column are features

    samples_train = X_train.shape[0] - look_back        # 4446-120=4326
    X_train_reshaped = np.zeros((samples_train, look_back, num_features))
    y_train_reshaped = np.zeros((samples_train))

    # 划分为三维张量
    for i in range(samples_train):
        y_position = i + look_back
        X_train_reshaped[i] = X_train[i:y_position]
        y_train_reshaped[i] = y_train[y_position]

    samples_test = X_test.shape[0] - look_back
    X_test_reshaped = np.zeros((samples_test, look_back, num_features))
    y_test_reshaped = np.zeros((samples_test))

    for i in range(samples_test):
        y_position = i + look_back
        X_test_reshaped[i] = X_test[i:y_position]
        y_test_reshaped[i] = y_test[y_position]
    
    return X_train_reshaped, y_train_reshaped, X_test_reshaped, y_test_reshaped

def downsample_results_new(x, y_pred, y_true, magnitude, resolution, model_name, path, savefig=False):
    """
    This function takes the hourly results and downsamples them to the given resolution.

    x, datetime values:test data
    y_pred, y predictions
    y_true, y true values
    magnitude, scaling factor for y axis = 1
    resolution, Pandas resample resolution e.g. 6H, D, W
    model_name, string containing model name, e.g. 'MVLR'
    savefig, wether to save the figure or not
    """

    datetime = x.index

    y_pred = y_pred.reshape(y_pred.shape[0])
    y_true = y_true.reshape(y_true.shape[0])

    results = pd.DataFrame(y_true, y_pred)
    result = results.reset_index()
    result.columns = ['y_pred', 'y_true']

    result['datetime'] = datetime
    result = result.set_index(['datetime'])

    result.to_csv(path + model_name + '_' + resolution + '_predictions.csv')
    result = result.resample(resolution).sum()
    result = result.dropna()

    ytrue = result['y_true']
    ypred = result['y_pred']
    n = len(result)

    # 计算评估指标
    mse_result = (1 / n) * np.sum((ypred - ytrue) ** 2)
    mape_result = (100 / n) * np.sum(np.abs((ytrue - ypred) / ypred))
    smape_result = (100 / n) * np.sum(np.abs((ytrue - ypred)) / (np.abs(ytrue) + np.abs(ypred)))

    # Create plot
    plt.figure(figsize=(20, 10))
    plt.plot(result.index, result['y_true'], '.-', color='red', label='Real values', alpha=0.5,
             ms=10)
    plt.plot(result.index, result['y_pred'], '.-', color='blue', label='Predicted values', ms=10)

    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates

    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.title(
        model_name + ' hourly predictions downsampled to ' + resolution + ' resolution. \n MSE = %.2f \n MAPE = %.1f [%%] \n SMAPE = %.1f [%%]' % (
        mse_result, mape_result, smape_result), fontsize=14)

    if savefig:
        plt.savefig(path + resolution + '_predict.png', dpi=1300)
    plt.close()
