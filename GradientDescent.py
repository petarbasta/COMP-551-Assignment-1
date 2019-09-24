import Preprocess
import CrossValidate

def fit(trainingSet, alpha):
    # TODO

def predict(weights, validationSet):
    # TODO

def evaluate_acc(actualValues, predictedValues):
    # TODO

# set a learning parameter alpha (5 is arbitrary)
alpha = 5

wineDataset = Preprocess.preprocessWine()
wineDatasets = CrossValidate.split(wineDataset, 5)

# GradientDescent class should have function "fit" with training set and some gradient descent parameter alpha as inputs
# GradientDescent class also has a predict and evaluate_acc function (see assignment pdf)
averageErrorWine = CrossValidate.kFoldCrossValidation(wineDatasets, (trainingSet) => fit(trainingSet, alpha), predict, evaluate_acc)