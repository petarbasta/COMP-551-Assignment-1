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
    n1 = len(class1)
    n0 = len(class0)
    prob1 = n1 / (n1 + n0)
    prob0 = 1 - prob1
    fmean0 = np.mean(class0, axis=0)
    fmean1 = np.mean(class1, axis=0)
    cov0 = np.cov(class0, rowvar=False, bias=False)
    cov1 = np.cov(class1, rowvar=False, bias=False)
    covar = (cov0*n0 + cov1*n1)/(n0+n1+2)


    return [prob0, prob1, fmean0, fmean1, covar]

def predict(validationSet, params):

    odds = []
    predictions = []
    bias = np.log(params[1]/params[0]) + 0.5*params[2].T @ np.linalg.inv(params[4]) @ params[2] - 0.5*params[3].T @ np.linalg.inv(params[4]) @ params[3]
    for row in validationSet:
        odds.append(bias + row[:-1]@np.linalg.inv(params[4])@(params[3]-params[2]))

    for number in odds:
        if number >= 0:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions

def evaluate_acc(actualValues, predictedValues):
    return np.sum(np.logical_and(actualValues[:,-1], predictedValues))/len(predictedValues)


wineDataset = Preprocess.preprocessWine()
wineDatasets = CrossValidate.split(wineDataset, 5)

averageErrorWine = CrossValidate.kFoldCrossValidation(wineDatasets, fit, predict, evaluate_acc)

tumourDataset = Preprocess.preprocessTumour()
tumourDatasets = CrossValidate.split(tumourDataset, 5)

averageErrorTumour = CrossValidate.kFoldCrossValidation(tumourDatasets, fit, predict, evaluate_acc)

print("Wine accuracy: " + str(averageErrorWine))
print("Tumor accuracy: " + str(averageErrorTumour))