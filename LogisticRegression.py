import Preprocess
import CrossValidate
import matplotlib
import numpy as np
import random
import time


class LogisticRegression:

    def __init__(self):
        self.wine_data = Preprocess.preprocessWine()
        self.wine_data_interact = LogisticRegression.interact(self.wine_data)
        # self.wine_data = self.wine_data_interact
        self.wine_vars = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'free sulfure dioxide',
                          'density', 'pH', 'sulphates', 'alcohol']
        self.wine_weights = []
        for i in range(0, len(self.wine_data[0])):
            self.wine_weights.append(0.0)
        self.wine_weights = np.array(self.wine_weights, dtype=float)


        self.cancer_data = Preprocess.preprocessTumour()
        self.cancer_vars = ['']
        self.cancer_weights = []
        for j in range(0, len(self.cancer_data[0])):
            self.cancer_weights.append(0.0)
        self.cancer_weights = np.array(self.cancer_weights, dtype=float)
        # print(list(enumerate(self.cancer_weights)))
        # print(self.cancer_data[0])
        # self.predict(self.cancer_weights, self.cancer_data[0][:-1], self.cancer_data[0][-1])

    @staticmethod
    def sigmoid(a):  # Sigmoid function from slide 17
        # print(1 / (1 + np.exp(-a)))
        return 1 / (1 + np.exp(-a))

    @staticmethod
    def interact(data_list):
        list_out = []
        for i in data_list:
            temp = i[:-1]
            # temp.app
            # temp = np.append(temp, i[10]*i[11])   # 5,8;6,8;6,10;9,11;10,10;10,11
            # temp = np.append(temp, i[5] * i[8])
            # temp = np.append(temp, i[6] * i[8])
            # temp = np.append(temp, i[6] * i[10])
            # temp = np.append(temp, i[9] * i[11])
            # temp = np.append(temp, i[10] * i[10])
            # temp = np.append(temp, i[10] * i[11])

            temp = np.append(temp, i[-1])
            list_out.append(temp)
        return np.array(list_out)

    def loss_fn(self):
        pass

    def weight_times_x(self, w, x):

        pass


    @staticmethod
    def learn_rate(iterations, data_length):
        return 1/1_000_000_000_000

    # @staticmethod
    # def train(data, weights, iterations):
    #     for x in data:
    #         for ct in range(iterations):
    #             weights = LogisticRegression.fit(weights, x[:-1], x[-1], LogisticRegression.learn_rate)
    #     return weights

    @staticmethod
    def linear_fn_eval(w, x):
        #  Implementing a = w0 + w1x1 + w2x2 + ... + wmxm from slide 18
        a_out = w[0]
        # add w0 above, eliminate w0 value from array below, so loop does not add last value to the sum
        for count, weight in np.ndenumerate(w[1:]):
            # print(count, weight, x[count])
            a_out = a_out + x[count] * weight
        return a_out

    @staticmethod
    def predict(w, x):
        a = LogisticRegression.linear_fn_eval(w, x)
        sigmoided = LogisticRegression.sigmoid(a)
        if sigmoided >= 0.5:
            return 1
        return 0

    @staticmethod
    def fit(w, x_in, learning_rate):
        to_add = 0
        w.astype(np.float)
        w_k1 = np.copy(w)
        for count in range(len(w)):
            for wt_count in range(len(x_in)):
                x_vals = x_in[wt_count][:-1]
                # indicator = LogisticRegression.predict(w, x_in[wt_count])
                indicator = x_in[wt_count][-1]
                inner_part = indicator - LogisticRegression.sigmoid(LogisticRegression.linear_fn_eval(w, x_in[wt_count]))
                to_add = to_add + learning_rate(count, len(x_in)) * x_in[wt_count][count] * inner_part  # Update Rule
            w_k1[count] = w[count] + to_add
            to_add = 0
        return w_k1

    # @staticmethod
    # def loss(h, y):
    #     return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


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

def get_folds(lr, data_in, k):
    # Setting up different folds
    data = data_in
    split_size = round(len(data) / k)
    learn_set = [[]] * k
    test_set = [[]] * k
    for count in range(k):
        learn_set[count] = []
        test_set[count] = []
    for i_k in range(k):
        for j in range(len(data)):
            if (j % k) == i_k:
                test_set[i_k].append(data[j].tolist())
            else:
                learn_set[i_k].append(data[j].tolist())
    return learn_set, test_set

def k_fold_cancer(lr):
    data_set = lr.cancer_data
    k = 5
    learn_l, test_l = get_folds(lr, data_set, k)
    start_t = time.time_ns()
    for i in range(k):
        # each iteration uses a different set of data
        learn = np.asarray(learn_l[i])
        test = np.asarray(test_l[i])

        lr.cancer_weights = LogisticRegression.fit(lr.cancer_weights, learn, LogisticRegression.learn_rate)
        y_act, y_pred = create_y_lists(test, lr.cancer_weights)
        print('CANCER: {}'.format(evaluate_acc(y_act, y_pred)))
    end_t = time.time_ns()
    total = end_t - start_t
    total = (total/5)/1_000_000_000
    print('Average Time CANCER: {}'.format(total))

def k_fold_wine(lr):
    data_set = lr.wine_data
    sum = 0
    k = 5
    learn_l, test_l = get_folds(lr, data_set, k)
    start_t = time.time_ns()
    for i in range(k):
        # each iteration uses a different set of data
        learn = np.asarray(learn_l[i])
        test = np.asarray(test_l[i])
        lr.wine_weights = LogisticRegression.fit(lr.wine_weights, learn, LogisticRegression.learn_rate)
        y_act, y_pred = create_y_lists(test, lr.wine_weights)
        ev = evaluate_acc(y_act, y_pred)
        print('WINE: {}'.format(ev))
        sum += ev
        # print(lr.wine_weights)
    end_t = time.time_ns()
    total = end_t - start_t
    total = (total/5)/1_000_000_000
    avg = sum / k
    print('AVG Time WINE: {}'.format(avg))
    # print('Average Time WINE: {}'.format(total))


def main():
    lr = LogisticRegression()
    k_fold_cancer(lr)
    k_fold_wine(lr)
    # y_act, y_pred = create_y_lists(lr.wine_data, lr.wine_weights)
    # print(evaluate_acc(y_act, y_pred))
    # num_iterations = 1000
    # for x in range(num_iterations):
    #     lr.wine_weights = LogisticRegression.fit(lr.wine_weights, lr.wine_data, LogisticRegression.learn_rate)
    #     y_act, y_pred = create_y_lists(lr.wine_data, lr.wine_weights)
    #     print(lr.wine_weights)
    #     print(evaluate_acc(y_act, y_pred))



if __name__ == "__main__":
    main()
