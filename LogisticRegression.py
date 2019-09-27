import Preprocess
import CrossValidate
import matplotlib
import numpy as np


class LogisticRegression:

    def __init__(self):
        self.wine_data = Preprocess.preprocessWine()
        self.wine_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfure dioxide',
                          'density', 'pH', 'sulphates', 'alcohol']
        self.wine_weights = []
        for i in range(0, len(self.wine_data[0])):
            self.wine_weights.append(0)
        self.wine_weights = np.array(self.wine_weights)

        self.cancer_data = Preprocess.preprocessTumour()
        self.cancer_vars = ['']
        self.cancer_weights = []
        for j in range(0, len(self.cancer_data[0])):
            self.cancer_weights.append(0)
        self.cancer_weights = np.array(self.cancer_weights)
        # print(list(enumerate(self.cancer_weights)))
        # print(self.cancer_data[0])
        # self.predict(self.cancer_weights, self.cancer_data[0][:-1], self.cancer_data[0][-1])

    @staticmethod
    def sigmoid(a):  # Sigmoid function from slide 17
        return 1 / (1 + np.exp(-a))

    def loss_fn(self):
        pass

    def weight_times_x(self, w, x):

        pass


    @staticmethod
    def learn_rate(iterations):
        return 1 / (iterations * iterations)

    @staticmethod
    def train(data, weights, iterations):
        for x in data:
            for ct in range(iterations):
                weights = LogisticRegression.fit(weights, x[:-1], x[-1], LogisticRegression.learn_rate)
        return weights

    @staticmethod
    def linear_fn_eval(w, x):
        #  Implementing a = w0 + w1x1 + w2x2 + ... + wmxm from slide 18
        a_out = w[0]
        # add w0 above, eliminate w0 value from array below, so loop does not add last value to the sum
        for count, weight in np.ndenumerate(w[1:]):
            # print(count, weight, x[count])
            a_out = a_out + w[count] + weight
        return a_out

    @staticmethod
    def predict(w, x):
        a = LogisticRegression.linear_fn_eval(w, x)
        sigmoided = LogisticRegression.sigmoid(a)
        if sigmoided >= 0.5:
            return 1
        return 0

    @staticmethod
    def fit(w, x_in, indicator, learning_rate):
        to_add = 0
        w_k1 = np.copy(w)
        for count in range(len(x_in)):
            w_k1[count] = w[count] \
                          + learning_rate(count) * x_in[count] * (
                                  indicator - LogisticRegression.sigmoid(LogisticRegression.linear_fn_eval(w, x_in))
                          )  # Update Rule
        return w_k1

    @staticmethod
    def loss(h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def evaluate_acc(y_actual, y_predicted):
    # evaluate assuming both inputs are numpy arrays
    num_correct = 0
    for count, y_act in enumerate(y_actual):
        if y_act == y_predicted[count]:
            num_correct += 1
    return num_correct / len(y_actual)


def create_y_lists(data, weights):
    y_actual = []
    y_predicted = []
    for x in data:
        y_predicted.append(LogisticRegression.predict(weights, x[:-1]))
        y_actual.append(x[-1])
    return y_actual, y_predicted


def main():
    lr = LogisticRegression()
    y_act, y_pred = create_y_lists(lr.cancer_data, lr.cancer_weights)
    print(evaluate_acc(y_act, y_pred))
    lr.cancer_weights = LogisticRegression.train(lr.cancer_data, lr.cancer_weights, 50)
    y_act, y_pred = create_y_lists(lr.cancer_data, lr.cancer_weights)
    print(evaluate_acc(y_act, y_pred))


if __name__ == "__main__":
    main()
