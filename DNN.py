import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, SGD, Nadam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import BatchNormalization
from keijzer import *
from keras.models import load_model
from sklearn.decomposition import PCA
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
magnitude = 1

# 完成数据切分和画图表示
def read_data(data_path, train_val_test_plot, available_data, train_test_plot,pca):
    df = pd.read_csv(data_path, delimiter='\t', parse_dates=['datetime'])
    datatime = df.iloc[:, 0]
    df = df.set_index(['datetime'])

    df.head()
    data = df.copy()
    y = data['gasPower']

    if pca:
        # 进行PCA降维
        modelPCA = PCA(n_components=0.9)  # 建立模型，设定保留主成分0.9
        X = np.array(data.iloc[:, 0:7])
        modelPCA.fit(X)
        X = modelPCA.transform(X)
        X = pd.DataFrame(X)
        X.insert(0, 'datatime', datatime)
        X = X.set_index(['datatime'])
        data = data.drop(df.columns[[0,1,2,3,4,5,6]],axis=1)
        data = X.join(data)

    columns_to_category = ['hour', 'dayofweek', 'season']
    data[columns_to_category] = data[columns_to_category].astype('category')

    data = pd.get_dummies(data, columns=columns_to_category)
    data.head()

    X = data.drop(['gasPower'], axis=1)
    train_size = 0.7
    val_size = 0.2

    split_index_val = int(data.shape[0] * (train_size - val_size))  # 0.5
    split_index_test = int(data.shape[0] * train_size)              # 0.7

    X_train = X[:split_index_val]
    X_test = X[split_index_test:]

    y_train = y[:split_index_val]
    y_val = y[split_index_val:split_index_test]
    y_test = y[split_index_test:]

    X_train_values = data[:split_index_val]
    X_val_values = data[split_index_val:split_index_test]
    X_test_values = data[split_index_test:]

    plt.figure(figsize=(20, 10))
    plt.plot(X_train_values.index, y_train, '.-', color='red', label='Train set', alpha=0.5)
    plt.close()
    plt.plot(X_val_values.index, y_val, '.-', color='orange', label='Validation set', alpha=0.5)
    plt.close()
    plt.plot(X_test_values.index, y_test, '.-', color='blue', label='Test set', alpha=0.5)
    plt.close()

    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.savefig(train_val_test_plot, dpi=1300)
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(data.index, data['gasPower'], '.-', color='red', label='Original data', alpha=0.5)
    plt.xlabel('Datetime [-]', fontsize=20)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % (magnitude), fontsize=14)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.title('Mean gas usage of 52 houses', fontsize=14)
    plt.tight_layout()
    plt.savefig(available_data, dpi=1200)
    plt.close()

    plt.figure(figsize=(20, 10))
    plt.plot(X_train_values.index, y_train, '.-', color='red', label='Train set', alpha=0.5)
    plt.plot(X_test_values.index, y_test, '.-', color='blue', label='Test set', alpha=0.5)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.savefig(train_test_plot, dpi=1300)
    plt.close()

    scalerX = StandardScaler()
    X_train = scalerX.fit_transform(X_train)
    X_test = scalerX.transform(X_test)
    return X_train, X_test, y_train, y_test, train_size, data

# 创建模型
def create_model(X_train):
    model = Sequential()
    model.add(Dense(256, input_shape=(X_train.shape[1],), kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.01))

    # 1
    for _ in range(1):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

    # 2
    for _ in range(1):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

    # 3
    for _ in range(1):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.01))
    # 4
    for _ in range(1):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

    # 5
    for _ in range(1):
        model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Dropout(0.01))

    # 6
    model.add(Dense(256, kernel_initializer='TruncatedNormal', use_bias=False))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Dropout(0.01))
    model.add(Dense(1))

    return model

def main():
    # 调整参数
    epochs = 1000
    bs = 128
    lr = 1e-3
    file = "rfe_knn_arima_pca"
    data_path = "../data/rfe-knn-arima.csv"
    train_val_test_plot = "../result/DNN/" + file + "/Train Val Test plot.png"
    available_data = "../result/DNN/" + file + "/available data.png"
    train_test_plot = "../result/DNN/" + file + "/Train Test plot.png"
    best_loss_hdf5 = "../result/DNN/" + file + "/DNN.best_loss.hdf5"
    besT_mape_hdf5 = "../result/DNN/" + file + "/DNN.best_mape.hdf5"
    predict_path = "../result/DNN/" + file + "/predict.png"
    ffd_path = "../result/DNN/" + file + "/Feedforward result hourly without dummy variables.png"
    path = "../result/DNN/" + file + '/'
    loss_png_path = "../result/DNN/" + file + "/DNN_loss.png"
    history_path = "../result/DNN/" + file + "/DNN_fit_history.csv"

    X_train, X_test, y_train, y_test, train_size, data = read_data(data_path, train_val_test_plot, available_data, train_test_plot, pca=True)

    adam = Adam(lr=lr)
    model = create_model(X_train)
    model.compile(loss=['mse'], metrics=[mape, smape, 'mse'], optimizer=adam)

    checkpoint1 = ModelCheckpoint(best_loss_hdf5, monitor='val_mape', verbose=1, save_best_only=True, mode='min')
    checkpoint2 = ModelCheckpoint(besT_mape_hdf5, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    early_stopping_monitor = EarlyStopping(patience=1000)
    result = model.fit(X_train, y_train, batch_size=bs, epochs=epochs, verbose=1, validation_split=0.2,
          callbacks=[early_stopping_monitor, checkpoint1, checkpoint2])
    print(model.summary())

    # plot loss
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
    model = load_model(best_loss_hdf5, custom_objects={'smape': smape, 'mape': mape})
    model.compile(loss='mse', metrics=[mape, smape], optimizer=adam)
    print('FINISHED')

    y_pred = model.predict(X_test)
    y_true = y_test.values.reshape(y_test.shape[0], 1)
    split_index = int(data.shape[0] * train_size)
    x = data[split_index:]

    plt.figure(figsize=(20, 10))
    plt.plot(x.index, y_true, '.-', color='red', label='Real values', alpha=0.5)
    plt.plot(x.index, y_pred, '.-', color='blue', label='Predicted values', alpha=1)
    plt.ylabel(r'gasPower $\cdot$ 10$^{-%s}$ [m$^3$/h]' % magnitude, fontsize=14)
    plt.xlabel('datetime [-]', fontsize=14)  # TODO: set x values as actual dates
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.legend(loc='upper left', borderaxespad=0, frameon=False, fontsize=14, markerscale=3)
    plt.savefig(predict_path)
    mse_result, mape_result, smape_result = model.evaluate(X_test, y_test)
    plt.title('DNN result \n MSE = %.2f \n MAPE = %.1f [%%] \n SMAPE = %.1f [%%]' % (mse_result, mape_result, smape_result),
              fontsize=14)
    plt.savefig(ffd_path, dpi=1200)
    print('FINISHED')
    plt.close()

    # 评估
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1H', model_name='DNN', path=path, savefig=True)
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1D', model_name='DNN', path=path, savefig=True)
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='1W', model_name='DNN', path=path, savefig=True)
    downsample_results_new(x, y_pred, y_true, magnitude=magnitude, resolution='4W', model_name='DNN', path=path, savefig=True)

if __name__=="__main__":
    main()

