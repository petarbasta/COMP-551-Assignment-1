import Preprocess
import CrossValidate

def fit(trainingSet, alpha):
    # TODO

def predict(weights, validationSet):
    # TODO

def evaluate_acc(actualValues, predictedValues):
    # TODO

wineDataset = Preprocess.preprocessWine()
wineDatasets = CrossValidate.split(wineDataset, 5)

# GradientDescent class should have function "fit" with training set and some gradient descent parameter alpha as inputs
# GradientDescent class also has a predict and evaluate_acc function (see assignment pdf)
averageErrorWine = CrossValidate.kFoldCrossValidation(wineDatasets, fit, predict, evaluate_acc)