import numpy as np # We will use np.ndarray as our array structure
import os      # We will use os.path to get the path to the current directory
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
# Get the directory of this file
# Get the full path to the data file
# Load data from the file using numpy "genfromtxt"
# Store all columns but the last one in features
# Store the last column in labels
# Return features and labels
def main(path):
    # Load train.csv into [Xtrain, Ytrain]
    Xtrain, Ytrain = load_data(path)
    return Xtrain, Ytrain

# calculates euclidean distance between two data points
def dist(a, b):
    tot = 0
    for i in range(len(b)):
        tot += (a[i] - b[i]) ** 2
    return math.sqrt(tot)

# returns an array of sorted indices of distances in ascending order
def sort_points(dists):
    return np.argsort(dists)

# calculates gaussian kernel for weights
def gaussian(sigma, dist):
    temp = 1/math.sqrt(2 * math.pi * (sigma ** 2))
    power = -1 * ((dist ** 2) / (2 * (sigma ** 2)))
    temp_two = math.pow(math.e, power)
    return temp * temp_two

# computes the weights for the distances using gaussian kernel
def get_weights(indices, dists, labels, k, sigma):
    weights = np.empty(k)
    sum = 0
    tot = 0
    for i in range(k):
        weights[i] = gaussian(sigma, dists[indices[i]])
        sum += weights[i]
    for i in range(k):
        tot += weights[i] / sum * labels[indices[i]]
    return tot

# calculates knn using the helper methods
def knn(features, labels, query, k, sigma):
    dists = np.empty(len(features))
    for i in range(len(features)):
        dists[i] = dist(features[i], query)
    indices = sort_points(dists)
    return get_weights(indices, dists, labels, k, sigma)

# calulates rmse
def rmse(labels, predictions):
    tot = 0
    for i in range(len(labels)):
        tot += (labels[i] - predictions[i]) ** 2
    return math.sqrt((1 / len(labels) * tot))

if __name__ == "__main__":
    # first load the train and test files into feature and label arrays
    features, labels = main('train.csv')
    queries, qlabels = main('test.csv')
    count = 0
    # next get the hyperparameters from hyper.csv
    f = open("hyper.csv", "r")
    k = int(f.readline())
    sigma = int(f.readline())
    f.close()
    # then calculate predictions of labels for test set using knn and output them to out.csv
    yHat = np.empty(len(queries))
    for i in range(len(queries)):
        yHat[i] = knn(features, labels, queries[i], k, sigma)
    np.savetxt(os.path.join(os.path.dirname(__file__), 'out.csv'), yHat, delimiter=',')
    # next open the out.csv file and calculate rmse based on the predictions of test.csv vs. the real labels
    file = open("out.csv", "r")
    predictions = np.empty(len(queries))
    while True:
        line = file.readline()
        if len(line) == 0:
            break
        predictions[count] = float(line)
        count += 1
    file.close()
    print(rmse(qlabels, predictions))