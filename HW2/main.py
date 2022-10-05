import numpy as np
import os
import math

def load_data(file_path: str)->tuple[np.ndarray, np.ndarray]:
    '''
    This function loads and parses text file separated by a ',' character and
    returns a data set as two arrays, an array of features, and an array of labels.
    Parameters
    ----------
    file_path : str
                path to the file containing the data set
    Returns
    -------
    features : ndarray
                2D array of shape (n,m) containing features for the data set
    labels : ndarray
                1D array of shape (n,) containing labels for the data set
'''
    curDirectory = os.path.dirname(__file__)
    fullPath = os.path.join(curDirectory, file_path)
    D = np.genfromtxt(fullPath, delimiter=",")
    features = D[:, :-1]
    labels = D[:, -1]
    return features, labels

def main(path):
    # Load train.csv into [Xtrain, Ytrain]
    Xtrain, Ytrain = load_data(path)
    return Xtrain, Ytrain

def normalize(features, test_features):
    for i in range(len(features[0])):
        max = -math.inf
        min = math.inf
        for arr in features:
            if(max < arr[i]):
                max = arr[i]
            if(min > arr[i]):
                min = arr[i]
        update(max, min, features, i)
        update(max, min, test_features, i)

def update(max, min, features, i):
    for arr in features:
        old = arr[i]
        temp = (old - min) / (max - min)
        new = 2 * temp - 1;
        arr[i] = new

def add_ones(features):
    end = len(features[0]) + 1
    new_arr = np.zeros((len(features), end))
    for i in range(len(features)):
        for j in range(len(features[i])):
            new_arr[i, j] = features[i, j]
        new_arr[i, end - 1] = 1
    return new_arr

def gradient_descent(features, labels, alpha, n_one, n_two):
    weight = np.zeros(len(features[0]))
    k = 0
    n = len(features)
    out_arr = np.zeros((n_one, len(features[0])))
    while k < n_two:
        if k < n_one:
            out_arr[k] = weight
        prediction = np.dot(features, weight)
        d_weight = (-1 / n) * alpha * features.T.dot((labels - prediction))
        weight = weight - d_weight
        k += 1
    return weight, out_arr

def rmse(labels, predictions):
    tot = (1/len(labels)) * sum(((labels - predictions) ** 2))
    return math.sqrt(tot)

if __name__ == "__main__":
    features_temp, labels = main('train.csv')
    t_features, t_labels = main('test.csv')
    normalize(features_temp, t_features)
    features = add_ones(features_temp)
    t_features = add_ones(t_features)
    f = open("hyper.csv", "r")
    lines = f.read().splitlines()
    alpha = float(lines[0])
    n_one = int(lines[1])
    n_two = int(lines[2])
    f.close()
    weights, out_array = gradient_descent(features, labels, alpha, n_one, n_two)
    np.savetxt(os.path.join(os.path.dirname(__file__), 'log.csv'), out_array, delimiter=',')
    t_predictions = np.dot(t_features, weights)
    root_mean = rmse(t_labels, t_predictions)
    w = open("out.csv", "w")
    w.write(str(root_mean))
    f.close()
