from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keijzer import *
from keras.models import load_model
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
magnitude = 1

def get_data(filename, pca):
    # Load the data：加载数据
    df = pd.read_csv(filename, delimiter='\t',parse_dates=['datetime'])
    # 抽取第一列
    datatime = df.iloc[:, 0]
    # 设置第一列为索引
    df = df.set_index(['datetime'])
    df.head()

    global data
    data = df.copy()
    y = data['gasPower']
    if pca:
        modelPCA = PCA(n_components=0.9)  # 建立模型，设定保留主成分0.9
        X = np.array(data.iloc[:, 0:7])
        modelPCA.fit(X)
        X = modelPCA.transform(X)
        X = pd.DataFrame(X)
        X.insert(0, 'datatime', datatime)
        X = X.set_index(['datatime'])
        data = data.drop(df.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
        data = X.join(data)

    # 时间信息转化为二进制编码
    columns_to_category = ['hour', 'dayofweek', 'season']
    data[columns_to_category] = data[columns_to_category].astype('category')

    data = pd.get_dummies(data, columns=columns_to_category)
    data.head()
    data.head()  # (6352,40)

    # 将数据转化成三维张量的形态
    # D -> 5, H -> 5*24=120,5天*24小时：时间步，LSTM根据过去120个小时的天然气使用量来预测未来下一个小时的使用量
    look_back = 5 * 24
    num_features = data.shape[1] - 1  # 39：特征个数
    train_size = 0.7

    # 这里得到的数据用于训练模型
    X_train, y_train, X_test, y_test = df_to_cnn_rnn_format(df=data, train_size=train_size, look_back=look_back,
                                                            target_column='gasPower', scale_X=True)
    return X_train, y_train, X_test, y_test, look_back, num_features, train_size

def plot_data(png_path):
    # Visualization of the train & test set target values
    plt.figure(figsize=(20, 10))

    # 画图
    plt.plot(data.index, data['gasPower'], '.-', color='red', label='Original data', alpha=0.5)
    # 设置label名
    plt.xlabel('Datetime [-]', fontsize=20)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % (magnitude), fontsize=14)

    # 确定坐标位置
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)

    # 设置题目
    plt.title('Mean gas usage of 52 houses', fontsize=14)
    plt.tight_layout()
    plt.savefig(png_path, dpi=1200)
    plt.close()

def plot_train_test_data(y_train,y_test,train_size,path):
    split_index = int(data.shape[0] * train_size)

    X_train_values = data[:split_index]  # get the datetime values of X_train,(4446,40)
    X_test_values = data[split_index:]  # get the datetime values of X_train,(1906,40)

    # 这里得到的数据用于画图
    datetime_difference = len(X_train_values) - len(y_train)  # 120
    X_train_values = X_train_values[datetime_difference:]

    datetime_difference = len(X_test_values) - len(y_test)
    X_test_values = X_test_values[
                    datetime_difference:]

    plt.figure(figsize=(20, 10))
    plt.plot(X_train_values.index, y_train, '.-', color='red', label='Train set', alpha=0.5)
    plt.plot(X_test_values.index, y_test, '.-', color='blue', label='Test set', alpha=0.5)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.savefig(path)
    plt.close()

# LSTM Model
def create_model(look_back,num_features):
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, num_features), return_sequences=False,
                   kernel_initializer='TruncatedNormal'))  # 8
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))  # 0.135

    # 全连接层多搭几层-解决非线性拟合的问题
    for _ in range(1):
        model.add(Dense(8, kernel_initializer='TruncatedNormal'))  # 128
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.3))  # 0.240

    for _ in range(1):
        model.add(Dense(8, kernel_initializer='TruncatedNormal'))  # 8
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.3))  # 0.636

    model.add(Dense(8, kernel_initializer='TruncatedNormal'))  # 16
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.3))  # 0.420
    model.add(Dense(1))

    return model

def main():
    epochs = 1000
    bs = 128
    lr = 1e-3
    file = "rfe_knn_arima"
    filename = "../data/rfe-knn-arima.csv"
    png_path = '../result/LSTM/' + file + '/available_data.png'
    val_mape_path = "../result/LSTM/" + file + "/LSTM.best.hdf5"
    val_loss_path = "../result/LSTM/" + file + "/LSTM.val_loss.hdf5"
    loss_png_path = '../result/LSTM/' + file + '/LSTM_loss.png'
    history_path = '../result/LSTM/' + file + '/LSTM_fit_history.csv'
    ffn_path = '../result/LSTM/' + file + '/Feedforward result hourly without dummy variables.png'
    path = '../result/LSTM/' + file + '/'
    data_path = '../result/LSTM/' + file + '/data.png'

    X_train, y_train, X_test, y_test, look_back, num_features, train_size = get_data(filename, False)

    plot_data(png_path)
    plot_train_test_data(y_train,y_test,train_size,data_path)
    adam = Adam(lr=lr)
    model = create_model(look_back, num_features)

    # compile & fit
    model.compile(optimizer=adam, loss=['mse'], metrics=[mape, smape, 'mse'])
    # # Fit the model
    early_stopping_monitor = EarlyStopping(patience=1000)

    # checkpoint
    checkpoint = ModelCheckpoint(val_mape_path, monitor='val_mape', verbose=1, save_best_only=True, mode='min')
    checkpoint1 = ModelCheckpoint(val_loss_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    result = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
                   verbose=1, callbacks=[early_stopping_monitor, checkpoint, checkpoint1])
    print(model.summary())  # 输出参数计算过程

    # 训练过程中的loss的变化过程
    plt.plot(result.history['loss'])
    plt.plot(result.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(loss_png_path)
    plt.close()
    pd.DataFrame(result.history).to_csv(history_path)

    # Load the architecture
    model = load_model(val_loss_path, custom_objects={'smape': smape, 'mape': mape})
    model.compile(loss='mse', metrics=[mape, smape], optimizer=adam)
    print('FINISHED')

    y_pred = model.predict(X_test)
    y_true = y_test.reshape(y_test.shape[0], 1)

    # 得到test数据集
    split_index = int(data.shape[0] * train_size)
    x = data[split_index:]
    datetime_difference = len(x) - len(y_true)  # 1786-1786=0
    x = x[datetime_difference:]

    plt.figure(figsize=(20, 10))
    plt.plot(x.index, y_true, '.-', color='red', label='Real values', alpha=0.5)
    plt.plot(x.index, y_pred, '.-', color='blue', label='Predicted values', alpha=1)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    mse_result, mape_result, smape_result = model.evaluate(X_test, y_test)
    plt.title(
    'LSTM result \n MSE = %.2f \n MAPE = %.1f [%%] \n SMAPE = %.1f [%%]' % (mse_result, mape_result, smape_result),
    fontsize=14)
    plt.savefig(ffn_path, dpi=1200)
    plt.close()
    print('FINISHED')

    # # Hour
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1H', model_name='LSTM', path=path, savefig=True)
    # # Day
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1D', model_name='LSTM', path=path, savefig=True)
    # # Week
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1W', model_name='LSTM', path=path, savefig=True)
    # # 4 Weeks
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='4W', model_name='LSTM', path=path, savefig=True)

if __name__=="__main__":
    main()