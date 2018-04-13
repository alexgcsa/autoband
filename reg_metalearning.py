__author__ = 'Silvia'
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os


def predict(x_train, x_test, y_train, y_test, i):
    # print 'predicting...', i, x_train.shape
    y_train.ravel()
    reg = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model = reg.fit(x_train, y_train)
    pred = model.predict(x_test)

    return pred[0]


def learner(target, meta_features, meta_features_test, target_test):
    predictions = []
    x_train = meta_features
    x_test = meta_features_test
    y_test = target_test


    for i in range(target.shape[1]):
        # print 'building test set...', i
        y_train = target[:, i]

        pr = predict(x_train, x_test, y_train, y_test, i)
        predictions.append(pr)

    # p = np.mean(predictions.values())

    # error = mean_squared_error(predictions.T, y_test)

    return predictions


def compile(auc):
    weights = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    w = []
    auc = auc[1:, :]
    for i in range(auc.shape[1]):
        a = np.asarray(auc[:, i])
        weights[a.argmax()] += 1
    # print weights

    for key, value in weights.items():
        # print weights[key]
        weights[key] = (weights[key]*100)/float(1200)
        w.append(weights[key])

    return w


def main(data_name):
    print("Starting metalearning...", data_name)

    dir = '/meta/auc/'
    m_features = '/meta/meta_features.csv'
    dir_list = os.listdir(dir)
    auc = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for e in dir_list:
        if e.endswith('.csv'):
            try:
                file_name = dir + os.path.basename(e)
                data = pd.read_csv(file_name)

                id = 0

                for i in range(data.shape[0]):
                    if data.loc[i, "name"] == data_name:
                        id = i
                        pass

                print(file_name)

                target = np.genfromtxt(file_name, delimiter=',')
                target = np.asmatrix(target[1:, 1:])
                target_test = target[id, :]
                target = np.delete(target, id, 0)

                meta_features = np.genfromtxt(m_features, delimiter=',')
                meta_features = np.asmatrix(meta_features[1:, 1:])
                meta_features_test = meta_features[id, :]
                meta_features = np.delete(meta_features, id, 0)

                r = learner(target, meta_features, meta_features_test, target_test)

                auc = np.vstack((auc, r))

                # print auc

            except ValueError:
               pass

    compile(auc)

    print('weights: ', compile(auc))

    print("Metalearning finished")
    return compile(auc)

