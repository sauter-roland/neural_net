import numpy as np
from scipy.special import expit


class NeuralNet:
    """
    Implementation of a complete neural net
    """

    def __init__(
            self,
            nodes,
            learning_rate=.3,
            init_kind='normal',
            activation_function=expit,
    ):
        self.input_nodes = nodes[0]
        self.output_nodes = nodes[2]
        self.hidden_nodes = nodes[1]

        self.learning_rate = learning_rate
        self.activation_function = activation_function

        self.init_weights_(kind=init_kind)

    def __str__(self):
        return_str = '\n'.join([
            f'Neural network with node setup: ({self.input_nodes}, {self.hidden_nodes}, {self.output_nodes})',
        ])
        return return_str

    def init_weights_random_(self):
        """
        Initializes weights to random floats between -0.5 and 0.5.
        """
        self.w_in_hidden = np.random.rand(self.hidden_nodes, self.input_nodes) - .5  # weights can be negative
        self.w_hidden_out = np.random.rand(self.output_nodes, self.hidden_nodes) - .5

    def init_weights_normal_(self):
        """
        Initializes weights to normally distributed floats.
        Standard deviation is root of connections in second layer.
        """
        self.w_in_hidden = np.random.normal(
            0.,  # mean of zero
            self.hidden_nodes ** (-.5),  # st.dev according to second layer
            (self.hidden_nodes, self.input_nodes)  # size of array
        )
        self.w_hidden_out = np.random.normal(
            0.,
            self.output_nodes ** (-.5),
            (self.output_nodes, self.hidden_nodes)
        )

    def init_weights_(self, kind='normal'):
        """
        Wrapper for the functions to initialize weights.
        """
        f_call = {
            'normal': self.init_weights_normal_,
            'random': self.init_weights_random_,
        }
        f_call[kind]()

    def query(self, val_in):
        """
        Takes an input and propagates it through the network
        """
        val_hidden = self.w_in_hidden @ val_in
        val_hidden = self.activation_function(val_hidden)
        val_out = self.w_hidden_out @ val_hidden
        val_out = self.activation_function(val_out)

        return val_out, val_hidden

    def predict(self, val_in):
        val_out = self.query(val_in)[0]
        return np.argmax(val_out)

    def train(self, data, targets):
        data = np.array(data, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        val_out, val_hidden = self.query(data)

        # calculate errors
        errors_out = targets - val_out
        errors_hidden = self.w_hidden_out.T @ errors_out

        # adjust weights
        self.w_hidden_out += self.learning_rate * (errors_out * val_out * (1. - val_out)) @ val_hidden.T
        self.w_in_hidden += self.learning_rate * (errors_hidden * val_hidden * (1. - val_hidden)) @ data.T

    def backprop_(self):
        pass


def test_init():
    nn = NeuralNet((2, 3, 2))
    shape_weights_in_hidden = nn.w_in_hidden.shape
    shape_weights_hidden_out = nn.w_hidden_out.shape
    assert(shape_weights_in_hidden == (3, 2))
    assert(shape_weights_hidden_out == (2, 3))


def test_query_shape():
    nn = NeuralNet((2,3,2))
    result = nn.query([1., 2.])
    assert(result[0].shape == (2,))
