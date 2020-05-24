from pathlib import Path
import numpy as np

from neural_net import NeuralNet
from data_prep import read_mnist


train_labels, train_features = read_mnist('./mnist_dataset/mnist_train_100.csv')
test_labels, test_features = read_mnist('./mnist_dataset/mnist_test_10.csv')

nn = NeuralNet((784, 100, 10))

for i in range(len(train_labels)):
    nn.train(train_features[i], train_labels[i])

results = [nn.predict(i) for i in test_features]
print(results)

print([np.argmax(i) for i in test_labels])
