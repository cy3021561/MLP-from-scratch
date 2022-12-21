import numpy as np
import pandas as pd
import sys
import time
import math
from matplotlib import pyplot as plt


class NN:
    def __init__(self, hidden_neuron, input_dim, batch_size, epoch, init_learning_rate, start_time):
        self.W1 = np.random.rand(hidden_neuron, input_dim)
        self.B1 = np.random.rand(hidden_neuron, 1)
        self.W2 = np.random.rand(2, hidden_neuron)
        self.B2 = np.random.rand(2, 1)
        self.hidden_neuron = hidden_neuron
        self.learning_rate = init_learning_rate
        self.batch_size = batch_size
        self.epoch = epoch
        self.train_array = []
        self.label_array = []
        self.loss_log = []
        self.validation_data = None
        self.validation_label = None
        self.start_time = start_time

    def extend_X(self, X):
        tmp_X = []
        for i in range(len(X)):
            x1, x2 = X[i][0], X[i][1]
            tmp = [x1, x2, math.sin(x1), math.sin(x2)]
            tmp_X.append(tmp)

        return np.array(tmp_X)

    def data_preprocess(self, raw_data, raw_label, validation_rate):
        data = np.array(raw_data)
        data = self.extend_X(data)
        label = np.array(raw_label)
        length, dim = data.shape
        all = np.concatenate((label, data), axis=1)
        np.random.shuffle(all)

        validation_data = all[int(length * (1 - validation_rate)):].T
        self.validation_label = validation_data[0]
        self.validation_data = validation_data[1:]

        train_data = all[:int(length * (1 - validation_rate))]
        self.train_array = []
        self.label_array = []
        for batch_num in range(len(train_data) // self.batch_size):
            s, e = batch_num * \
                self.batch_size, (batch_num + 1) * self.batch_size
            tmp_data = train_data[s:e].T
            tmp_y = tmp_data[0]
            tmp_x = tmp_data[1:]
            self.train_array.append(tmp_x)
            self.label_array.append(tmp_y)
        return

    def tanH(self, Z):
        return np.tanh(Z)

    def de_tanH(self, Z):
        return 1 - Z ** 2

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def sigmoid_derivative(self, X):
        sig = self.sigmoid(X)
        return np.multiply(sig, (1 - sig))

    def oneHot(self, Y):
        one_hot_y = np.zeros((Y.size, 2))
        for i in range(Y.size):
            if Y[i] == 0.0:
                one_hot_y[i][0] = 1
            else:
                one_hot_y[i][1] = 1
        return one_hot_y.T

    def softmax(self, Z):
        exp = np.exp(Z - np.max(Z))
        return exp / exp.sum(axis=0)

    def cross_entropy(self, truth_labels, outputs):
        loss = -np.sum(truth_labels * np.log(outputs))
        return loss / float(outputs.shape[0])

    def foward(self, X):
        Z1 = self.W1.dot(X) + self.B1
        A1 = self.sigmoid(Z1)
        Z2 = self.W2.dot(A1) + self.B2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backprop(self, Z1, A1, Z2, A2, Y, X):
        one_hot_y = self.oneHot(Y)
        dZ2 = 2 * (A2 - one_hot_y)
        dW2 = self.batch_size * (dZ2.dot(A1.T))
        dB2 = 1 / self.batch_size * np.sum(dZ2, 1)
        dZ1 = self.W2.T.dot(dZ2) * self.sigmoid_derivative(Z1)
        dW1 = 1 / self.batch_size * (dZ1.dot(X.T))
        dB1 = 1 / self.batch_size * np.sum(dZ1, 1)
        return dW1, dB1, dW2, dB2

    def update_paras(self, dW1, dB1, dW2, dB2):
        self.W1 = self.W1 - self.learning_rate * dW1
        self.B1 = self.B1 - self.learning_rate * \
            np.reshape(dB1, (self.hidden_neuron, 1))
        self.W2 = self.W2 - self.learning_rate * dW2
        self.B2 = self.B2 - self.learning_rate * np.reshape(dB2, (2, 1))
        return

    def get_predictions(self, A2):
        return np.argmax(A2, 0)

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y)/Y.size

    def train(self):
        for e in range(self.epoch):
            cur_time = time.time()
            if (cur_time - self.start_time) > 100:
                break
            self.loss_log = []
            for X, Y in zip(self.train_array, self.label_array):
                Z1, A1, Z2, A2 = self.foward(X)
                one_hot_y = self.oneHot(Y)
                loss = self.cross_entropy(one_hot_y, A2)
                self.loss_log.append(loss)
                dW1, dB1, dW2, dB2 = self.backprop(Z1, A1, Z2, A2, Y, X)
                self.update_paras(dW1, dB1, dW2, dB2)
            # Validation
            Z1, A1, Z2, out_A2 = self.foward(self.validation_data)
            predictions = self.get_predictions(out_A2)
            # print("Val acc: ", self.get_accuracy(
            #     predictions, self.validation_label))

            # print("Epoch: ", e, " training loss: ",
            #       np.mean(np.array(self.loss_log)))
        return

    def out_put(self, predictions):
        np.savetxt('test_predictions.csv', predictions, fmt="%d")
        return

    def test(self, test_data_dir):  # , test_label_dir):
        raw_data = pd.read_csv(test_data_dir, header=None)
        #raw_label = pd.read_csv(test_label_dir, header=None)
        data = np.array(raw_data)
        data = self.extend_X(data)
        #label = np.array(raw_label)
        #all = np.concatenate((label, data), axis=1)
        # test_data = all.T
        # Y = test_data[0]
        # X = test_data[1:]
        X = data.T
        Z1, A1, Z2, A2 = self.foward(X)
        predictions = self.get_predictions(A2)
        self.out_put(predictions)
        # print(predictions)
        # print("Test acc: ", self.get_accuracy(
        #     predictions, Y))

        return


if __name__ == '__main__':
    # circle_train_data.csv spiral_train_data.csv gaussian_train_data.csv ./data/xor_train_data.csv
    start_time = time.time()
    TRAIN_DATA = sys.argv[1]  # './data/xor_train_data.csv'
    TRAIN_LABEL = sys.argv[2]  # './data/xor_train_label.csv'
    TEST_DATA = sys.argv[3]  # './data/xor_test_data.csv'
    # TEST_LABEL = sys.argv[4]  # './data/xor_test_label.csv'
    HIDDEN_NEURON = 8
    INIT_LEARNING_RATE = 0.03
    VALIDATION_RATE = 0.1
    BATCH_SIZE = 32
    EPOCH = 100000

    raw_data = pd.read_csv(TRAIN_DATA, header=None)
    raw_label = pd.read_csv(TRAIN_LABEL, header=None)
    mlp = NN(HIDDEN_NEURON, 4, BATCH_SIZE, EPOCH,
             INIT_LEARNING_RATE, start_time)
    mlp.data_preprocess(raw_data, raw_label, VALIDATION_RATE)
    mlp.train()
    mlp.test(TEST_DATA)  # , TEST_LABEL)
