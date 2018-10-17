import numpy as np

import backend

class Perceptron(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.get_data_and_monitor = backend.make_get_data_and_monitor_perceptron()

        "*** YOUR CODE HERE ***"
        # initialize an array of zeros with dimensions specified
        self.weights = np.zeros(dimensions)

    def get_weights(self):
        """
        Return the current weights of the perceptron.

        Returns: a numpy array with D elements, where D is the value of the
            `dimensions` parameter passed to Perceptron.__init__
        """
        "*** YOUR CODE HERE ***"
        # return the seld.weights
        return self.weights

    def predict(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # if the dot product of x and self.weights are non-negative, return 1
        # else -1
        if np.dot(x, self.weights) >= 0:
            return 1
        else:
            return -1

    def update(self, x, y):
        """
        Update the weights of the perceptron based on a single example.
            x is a numpy array with D elements, where D is the value of the
                `dimensions`  parameter passed to Perceptron.__init__
            y is either 1 or -1

        Returns:
            True if the perceptron weights have changed, False otherwise
        """
        "*** YOUR CODE HERE ***"
        # there are two cases that we need to update the weights
        # 1. if dot product of self.weights and x are non-negative but y = -1
        # 2. if dot product of self.weights and x are negative but y = 1
        if self.predict(x) * y < 0:
            self.weights += y * x
            return True
        return False

        # if np.dot(self.weights, x) >= 0 and y < 0:
        #     self.weights -= x
        #     return True
        # if np.dot(self.weights, x) < 0 and y > 0:
        #     self.weights += x
        #     return True
        # return False


    def train(self):
        """
        Train the perceptron until convergence.

        To iterate through all of the data points once (a single epoch), you can
        do:
            for x, y in self.get_data_and_monitor(self):
                ...

        get_data_and_monitor yields data points one at a time. It also takes the
        perceptron as an argument so that it can monitor performance and display
        graphics in between yielding data points.
        """
        "*** YOUR CODE HERE ***"
        check = True
        while check:
            check = False
            # in the for loop, if one update happens, check will become True
            # and will lead to another loop for all data
            for x, y in self.get_data_and_monitor(self):
                check = check or self.update(x, y)










