import numpy as np

# splits dataset into k lists
def split(dataset, k):
    x, y = divmod(len(dataset), k)
    datasets = []
    for i in range(k):
        datasets.append(dataset[i * x + min(i, y):(i + 1) * x + min(i + 1, y)])
    
    return datasets

# runs k fold cross validation
def kFoldCrossValidation(datasets, fit, predict, evaluate_acc):
    errors = []

    # for each dataset
    for i in range(len(datasets)):
        trainingSet = []
        validationSet = []

        for j in range(len(datasets)):
            if (i != j):
                # put everything that isn't the current dataset into the training set
                list.extend(trainingSet, datasets[j])
            else:
                # set the validation set to the current dataset
                validationSet = datasets[j]
        
        # train the model
        weights = fit(trainingSet)

        # predict the outputs
        predictedValues = predict(validationSet, weights)

        # check how accurate the model is
        errors.append(evaluate_acc(validationSet, predictedValues))
    
    #return the average of the errors
    return np.mean(errors)