import Preprocess
import CrossValidate
import numpy as np

def fit(trainingSet, alpha=0):

    n1 = np.sum(trainingSet[:,-1])
    n0 = trainingSet.shape[1] - n1
    prob1 = n1 / trainingSet.shape[1]
    prob0 = 1 - prob1

    sum0 = np.zeros(trainingSet.shape[0] - 1)
    sum1 = np.zeros(trainingSet.shape[0] - 1)
    for row in trainingSet:
        if row[-1]:
            sum1 = np.add(sum1,row[:-1])
        else:
            sum0 = np.add(sum0,row[:-1])
    fmean0 = sum0 / n0
    fmean1 = sum1 / n1

    covar = np.zeros((fmean1.shape[0], fmean1.shape[0]))
    for row in trainingSet:
        if row[-1]:
            covar += (row[:-1] - fmean1) * np.transpose(row[:-1] - fmean1)
        else:
            covar += (row[:-1] - fmean0) * np.transpose(row[:-1] - fmean0)
    covar = covar / (n1 + n0 - 2)

    return [prob0, prob1, fmean0, fmean1, covar]

def predict(validationSet, params):

    predictions = []
    for row in validationSet:
        predictions.append(
            np.log(params[0]/params[1])
            + 0.5*np.multiply(params[2],np.multiply(np.transpose(params[2]),params[4]))
            - 0.5*np.multiply(params[3],np.multiply(np.transpose(params[3]),params[4]))
            + np.multiply(np.multiply(np.transpose(validationSet[:-1]),params[4]),(params[3] - params[2]))
        )

    for odds in predictions:
        if odds >= 0:
            odds = 1
        else:
            odds = 0

    return  predictions

def evaluate_acc(actualValues, predictedValues):
    return ((actualValues==1 and predictedValues==1) or (actualValues==0 and predictedValues==0)).sum()/actualValues.len()

wineDataset = Preprocess.preprocessWine()
wineDatasets = CrossValidate.split(wineDataset, 5)

# GradientDescent class should have function "fit" with training set and some gradient descent parameter alpha as inputs
# GradientDescent class also has a predict and evaluate_acc function (see assignment pdf)
averageErrorWine = CrossValidate.kFoldCrossValidation(wineDatasets, fit, predict, evaluate_acc)

tumourDataset = Preprocess.preprocessTumour()
tumourDatasets = CrossValidate.split(tumourDataset, 5)

averageErrorTumour = CrossValidate.kFoldCrossValidation(tumourDatasets, fit, predict, evaluate_acc)