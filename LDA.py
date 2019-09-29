import Preprocess
import CrossValidate
import numpy as np

def fit(trainingSet, alpha=0):

    class1 = []
    class0 = []
    for row in trainingSet:
        if row[-1] == 1:
            class1.append(row[:-1])
        else:
            class0.append(row[:-1])

    # Number of instances of each class
    n1 = len(class1)
    n0 = len(class0)

    # Sample probability of each class
    prob1 = n1 / (n1 + n0)
    prob0 = 1 - prob1

    # Feature means for each class
    fmean0 = np.mean(class0, axis=0)
    fmean1 = np.mean(class1, axis=0)

    # Covariance matrix
    cov0 = np.cov(class0, rowvar=False, bias=False)
    cov1 = np.cov(class1, rowvar=False, bias=False)
    covar = (cov0*n0 + cov1*n1)/(n0+n1+2)

    return [prob0, prob1, fmean0, fmean1, covar]
    # Previously implemented calculating the feature means and covariance manually but found out
    # we were allowed to use numpy functions so now it looks like this.

def predict(validationSet, params):

    odds = []
    predictions = []

    # Bias term
    bias =  np.log(params[1]/params[0]) + 0.5*params[2].T @ np.linalg.inv(params[4]) @ params[2] - 0.5*params[3].T @ np.linalg.inv(params[4]) @ params[3]

    # Calculating discriminant term
    for row in validationSet:
        odds.append(bias + row[:-1]@np.linalg.inv(params[4])@(params[3]-params[2]))

    # Converting log odds to binary class
    for number in odds:
        if number >= 0:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

# Calculating accuracy
def evaluate_acc(actualValues, predictedValues):
    num_correct = 0
    for count, y_act in enumerate(actualValues[:,-1]):
        if y_act == predictedValues[count]:
            num_correct += 1
    return num_correct / len(predictedValues)

# Setting up wine data sets
wineDataset = Preprocess.preprocessWine()
wineDatasets = CrossValidate.split(wineDataset, 5)

# Testing wine with 1 feature removed
for i in range(0,wineDataset.shape[1] - 1):
    newSet = []
    for dataset in wineDatasets:
        newData = []
        for row in dataset:

           newData.append(np.delete(row, i))
        newSet.append(np.array(newData))

    averageErrorWine = CrossValidate.kFoldCrossValidation(np.array(newSet), fit, predict, evaluate_acc)
    print("Wine accuracy without (" + str(i) + "): " + str(averageErrorWine))

# Testing with including the (i*j)th interaction term
for i in range(0,wineDataset.shape[1] - 1):
    for j in range(0,wineDataset.shape[1] - 1 - i):

        newSet = []
        for dataset in wineDatasets:

            newData = []
            for row in dataset:
                newData.append(np.insert(row, 0, row[i]*row[j]))

            newSet.append(np.array(newData))

        averageErrorWine = CrossValidate.kFoldCrossValidation(np.array(newSet), fit, predict, evaluate_acc)
        print("Wine accuracy with (" + str(i) + "," + str(j) + "): " + str(averageErrorWine))

# Testing wine with unmodified dataset
averageErrorWine = CrossValidate.kFoldCrossValidation(wineDatasets, fit, predict, evaluate_acc)
print("Wine accuracy: " + str(averageErrorWine))

# Setting up tumor data
tumourDataset = Preprocess.preprocessTumour()
tumourDatasets = CrossValidate.split(tumourDataset, 5)

# Testing tumors with 1 feature removed
for i in range(0,tumourDataset.shape[1] - 1):
    newSet = []
    for dataset in tumourDatasets:
        newData = []
        for row in dataset:

           newData.append(np.delete(row, i))
        newSet.append(np.array(newData))

    averageErrorTumour = CrossValidate.kFoldCrossValidation(np.array(newSet), fit, predict, evaluate_acc)
    print("Tumour accuracy without (" + str(i) + "): " + str(averageErrorTumour))

# Testing tumors including the (i*j)th interaction term
for i in range(0,tumourDataset.shape[1] - 1):
    for j in range(0,tumourDataset.shape[1] - 1 - i):

        newSet = []
        for dataset in tumourDatasets:

            newData = []
            for row in dataset:
                newData.append(np.insert(row, 0, row[i]*row[j]))

            newSet.append(np.array(newData))

        averageErrorTumour = CrossValidate.kFoldCrossValidation(np.array(newSet), fit, predict, evaluate_acc)
        print("Tumor accuracy with (" + str(i) + "," + str(j) + "): " + str(averageErrorTumour))

# Testing unmodified tumor data set
averageErrorTumour = CrossValidate.kFoldCrossValidation(tumourDatasets, fit, predict, evaluate_acc)
print("Tumor accuracy: " + str(averageErrorTumour))