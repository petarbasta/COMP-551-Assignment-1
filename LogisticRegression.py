import Preprocess
import CrossValidate
import matplotlib
import numpy as np

class LogisticRegression:

    def __init__(self):
        self.wine_data = Preprocess.preprocessWine()
        self.wine_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfure dioxide', 'density', 'pH', 'sulphates', 'alcohol']
        self.wine_weights = []

        self.cancer_data = Preprocess.preprocessTumour()
        self.cancer_vars = ['']
        print(self.cancer_data[0])
        pass

    @staticmethod
    def logistic_fn(a):
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

def main():
    logreg = LogisticRegression()

if __name__ == "__main__":
    main()