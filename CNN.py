from keras.models import Sequential
from keras.layers import Dense,  Flatten, Dropout,  Conv2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keijzer import *
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from sklearn.decomposition import PCA
magnitude = 1

def get_data(filename,pca):
    # Load the data
    df = pd.read_csv(filename, delimiter='\t', parse_dates=['datetime'])
    datatime = df.iloc[:, 0]
    df = df.set_index(['datetime'])

    df.head()
    global data
    data = df.copy()
    if pca:
        # PCA降维，设定保留主成分0.9
        modelPCA = PCA(n_components=0.9)

        X = np.array(data.iloc[:, 0:7])
        modelPCA.fit(X)
        X = modelPCA.transform(X)
        X = pd.DataFrame(X)
        X.insert(0, 'datatime', datatime)
        X = X.set_index(['datatime'])

        data = data.drop(df.columns[[0, 1, 2, 3, 4, 5, 6]], axis=1)
        data = X.join(data)

    # Datetime info to categorical
    columns_to_category = ['hour', 'dayofweek', 'season']
    data[columns_to_category] = data[columns_to_category].astype('category')  # change datetypes to category

    data = pd.get_dummies(data, columns=columns_to_category)
    data.head()

    # D -> 5, H -> 5*24=120,5天*24小时：时间步，LSTM根据过去120个小时的天然气使用量来预测未来下一个小时的使用量
    look_back = 5 * 24
    num_features = data.shape[1] - 1  # 39：特征个数
    train_size = 0.7

    # 这里得到的数据用于训练模型
    X_train, y_train, X_test, y_test = df_to_cnn_rnn_format(df=data, train_size=train_size, look_back=look_back,
                                                            target_column='gasPower', scale_X=True)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    return X_train, y_train, X_test, y_test, look_back, num_features, train_size

def plot_data(png_path):
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
    datetime_difference = len(X_train_values) - len(y_train)
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

seed = 42
def create_model(X_train):
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = Sequential()

    ks1_first = 3
    ks1_second = 3
    ks2_first = 3
    ks2_second = 3
    model.add(Conv2D(filters=(3),
                     kernel_size=(ks1_first, ks1_second),
                     input_shape=input_shape,
                     padding='same',
                     kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.025))

    for _ in range(1):
        model.add(Conv2D(filters=(3),
                         kernel_size=(ks2_first, ks2_second),
                         padding='same',
                         kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    model.add(Flatten())

    for _ in range(4):
        model.add(Dense(64, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    for _ in range(3):
        model.add(Dense(64, kernel_initializer='TruncatedNormal'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.1))

    model.add(Dense(64, kernel_initializer='TruncatedNormal'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.1))

    model.add(Dense(1))

    return model

def main():
    epochs = 1000
    bs = 128
    lr = 1e-3
    file = "rfe_knn"
    filename = "../data/rfe-knn.csv"
    png_path = "../result/CNN/" + file + "/available_data.png"
    val_mape_path = "../result/CNN/" + file + "/CNN.best.hdf5"
    val_loss_path = "../result/CNN/" + file + "/CNN.val_loss.hdf5"
    loss_png_path = "../result/CNN/" + file + "/CNN_loss.png"
    history_path = "../result/CNN/" + file + "/CNN_fit_history.csv"
    ffn_path = "../result/CNN/" + file + '/Feedforward result hourly without dummy variables.png'
    path = "../result/CNN/" + file + '/'
    data_path = "../result/CNN/" + file + '/data.png'

    X_train, y_train, X_test, y_test, look_back, num_features, train_size = get_data(filename, pca=False)
    plot_data(png_path)
    plot_train_test_data(y_train, y_test, train_size, data_path)

    model = create_model(X_train)
    adam = Adam(lr=lr)

    # compile & fit
    model.compile(loss=['mse'], metrics=[mape, smape, 'mse'], optimizer=adam)
    early_stopping_monitor = EarlyStopping(patience=1000)
    checkpoint = ModelCheckpoint(val_mape_path, monitor='val_mape', verbose=1, save_best_only=True, mode='min')
    checkpoint1 = ModelCheckpoint(val_loss_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    result = model.fit(X_train, y_train, epochs=epochs, batch_size=bs, validation_split=0.2,
          verbose=1, callbacks=[early_stopping_monitor, checkpoint, checkpoint1])
    print(model.summary())

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
    datetime_difference = len(x) - len(y_true)
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
        'CNN result \n MSE = %.2f \n MAPE = %.1f [%%] \n SMAPE = %.1f [%%]' % (mse_result, mape_result, smape_result),
        fontsize=14)
    plt.savefig(ffn_path, dpi=1200)
    plt.close()
    print('FINISHED')

    # Hour
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1H', model_name='CNN', path=path, savefig=True)
    # Day
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1D', model_name='CNN', path=path, savefig=True)
    # Week
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1W', model_name='CNN', path=path, savefig=True)
    # 4 Weeks
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='4W', model_name='CNN', path=path, savefig=True)

if __name__ == "__main__":
    main()