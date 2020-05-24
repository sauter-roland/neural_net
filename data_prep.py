from pathlib import Path
import numpy as np
import pytest

def read_mnist(file_path):
    with open(Path(file_path)) as f:
        data = f.read()
    data = data.split('\n')
    data = [array.split(',') for array in data]
    labels = []
    features = []    
    for array in data:
        if array[0]:
            current_label = int(array[0])
            current_features = np.asfarray(array[1:])
            current_features.reshape((28, 28))
            current_features = (current_features / 255 * .99) + .01
            label_array = np.zeros(10) + .01
            label_array[current_label] = .99
            
            labels.append(label_array)
            features.append(current_features)
    return labels, features

@pytest.fixture
def setup():
    output = read_mnist(
        './mnist_dataset/mnist_test_10.csv'
        )
    return output

def test_number_of_labels(setup):
    labels = setup[0]
    assert(labels[0].size == 10)

def test_label_values(setup):
    labels = setup[0]
    assert(labels[0][7] > .9)

def test_feature_dimensions(setup):
    features = setup[1]
    assert(features[0].size == 28 * 28)
