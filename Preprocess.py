import numpy as np

# import data from file
def loadDataSet(fileName, delimiter):
    return np.genfromtxt(fileName, delimiter=delimiter)

# sets last column to 0 or 1 based on threshold
def binarizeData(dataset, threshold):
    size = dataset.shape[1] - 1

    for example in dataset:
        if example[size] > threshold:
            example[size] = 1
        else:
            example[size] = 0

    return dataset

# sets mean to 0 and standard deviation to 1
def standardizeData(dataset):
    means = np.mean(dataset, axis=0)
    standardDeviations = np.std(dataset, axis=0)

    for example in dataset:
        for index, feature in enumerate(example[:-1]):
            example[index] = (feature - means[index]) * 1 / standardDeviations[index]

# converts data to workable form
def preprocessWine():
    wineDataset = loadDataSet('winequality-red.csv', ';')
    standardizeData(wineDataset)
    binarizeData(wineDataset, 5)
    
    return wineDataset

def preprocessTumour():
    tumourDataset = loadDataSet('breast-cancer-wisconsin.data', ',')
    #remove first ID column
    tumourDataset = tumourDataset[:,1:]
    standardizeData(tumourDataset)
    binarizeData(tumourDataset, 3)

    return tumourDataset    