import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        `get_data_and_monitor` will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to `get_data_and_monitor`, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.2
        # hidden layer size
        self.h = 200
        # set parameters for this question
        batch_size = 0
        for x, y in self.get_data_and_monitor(self):
            batch_size, junk = x.shape
            break
        self.W1 = nn.Variable(self.h, batch_size)
        self.W2 = nn.Variable(batch_size, self.h)
        self.b1 = nn.Variable(self.h, 1)
        self.b2 = nn.Variable(batch_size, 1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # set the trainable variables here
        # in the first try, we will do 2 layers
        # f(x) = W2 * ReLU(W1 * x + b1) + b2
        # size of each variable:
        # x: i * 1
        # W1: h * i, b1: h * 1
        # W2: i * h, b2: i * 1

        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2])
        input_x = nn.Input(graph, x)
        mul_1 = nn.MatrixMultiply(graph, self.W1, input_x)
        add_1 = nn.MatrixVectorAdd(graph, mul_1, self.b1)
        # add_1 = nn.Add(graph, mul_1, self.b1)
        relu_1 = nn.ReLU(graph, add_1)
        mul_2 = nn.MatrixMultiply(graph, self.W2, relu_1)
        add_2 = nn.Add(graph, mul_2, self.b2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, add_2, input_y)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(add_2)

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        # hidden layer size
        self.h = 20
        # set parameters for this question
        self.batch_size = 200
        self.W1 = nn.Variable(1, self.h)
        self.W2 = nn.Variable(self.h, 1)
        self.b1 = nn.Variable(1, self.h)
        self.b2 = nn.Variable(self.batch_size, 1)

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs `y` are known during training, but not at test time.
        If correct outputs `y` are provided, this method must construct and
        return a nn.Graph for computing the training loss. If `y` is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "*** YOUR CODE HERE ***"
        # basically the same as last question
        # f(x) = g(x) - g(-x)
        # g(x) = ReLU(x * W1 + b1) * W2 + b2
        # size of each variable:
        # x: i * 1
        # W1: i * h, b1: 1 * h
        # W2: h * 1, b2: i * 1
        # print x.shape = 16, 1
        batch_size, junk = x.shape
        if not batch_size == self.batch_size:
            self.batch_size = batch_size
            self.W1 = nn.Variable(1, self.h)
            self.W2 = nn.Variable(self.h, 1)
            self.b1 = nn.Variable(1, self.h)
            self.b2 = nn.Variable(self.batch_size, 1)
        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2])
        input_x = nn.Input(graph, x)
        mul_1 = nn.MatrixMultiply(graph, input_x, self.W1)
        add_1 = nn.MatrixVectorAdd(graph, mul_1, self.b1)
        relu_1 = nn.ReLU(graph, add_1)
        mul_2 = nn.MatrixMultiply(graph, relu_1, self.W2)
        add_2 = nn.Add(graph, mul_2, self.b2)

        negation = np.array([[-1.]])
        n_x = np.dot(x, negation)
        input_n_x = nn.Input(graph, n_x)
        n_mul_1 = nn.MatrixMultiply(graph, input_n_x, self.W1)
        n_add_1 = nn.MatrixVectorAdd(graph, n_mul_1, self.b1)
        n_relu_1 = nn.ReLU(graph, n_add_1)
        n_mul_2 = nn.MatrixMultiply(graph, n_relu_1, self.W2)
        n_add_2 = nn.Add(graph, n_mul_2, self.b2)
        # at last, we need to do a negation here
        negation_node = nn.Input(graph, negation)
        n_n_add_2 = nn.MatrixMultiply(graph, n_add_2, negation_node)
        value_n_n_add_2 = graph.get_output(n_n_add_2)
        f = nn.Add(graph, add_2, n_n_add_2)

        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, f, input_y)
            return graph

        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            "*** YOUR CODE HERE ***"
            return graph.get_output(f)

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 2.0
        # hidden layer size
        self.h1 = 400
        self.h2 = 300
        self.h3 = 300
        self.h4 = 300
        # set parameters for this question
        self.W1 = nn.Variable(784, self.h1)
        self.b1 = nn.Variable(1, self.h1)
        self.W2 = nn.Variable(self.h1, self.h2)
        self.b2 = nn.Variable(1, self.h2)
        self.W3 = nn.Variable(self.h2, self.h3)
        self.b3 = nn.Variable(1, self.h3)
        self.W4 = nn.Variable(self.h3, self.h4)
        self.b4 = nn.Variable(1, self.h4)
        self.W5 = nn.Variable(self.h4, 10)
        self.b5 = nn.Variable(1, 10)
        

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"

        # batch_size, junk = x.shape
        # if not batch_size == self.batch_size:
        #     self.batch_size = batch_size
        #     self.W1 = nn.Variable(784, self.h1)
        #     self.W2 = nn.Variable(self.h1, 10)
        #     self.b1 = nn.Variable(1, self.h1)
        #     self.b2 = nn.Variable(1, 10)
            # self.b2 = nn.Variable(self.batch_size, 1)
        # f(x) = ReLU(x * W1 + b1) * W2 + b2
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4, self.W5, self.b5])
        input_x = nn.Input(graph, x)

        mul_1 = nn.MatrixMultiply(graph, input_x, self.W1)
        add_1 = nn.MatrixVectorAdd(graph, mul_1, self.b1)
        relu_1 = nn.ReLU(graph, add_1)

        mul_2 = nn.MatrixMultiply(graph, relu_1, self.W2)
        add_2 = nn.MatrixVectorAdd(graph, mul_2, self.b2)
        relu_2 = nn.ReLU(graph, add_2)

        mul_3 = nn.MatrixMultiply(graph, relu_2, self.W3)
        add_3 = nn.MatrixVectorAdd(graph, mul_3, self.b3)
        relu_3 = nn.ReLU(graph, add_3)

        mul_4 = nn.MatrixMultiply(graph, relu_3, self.W4)
        add_4 = nn.MatrixVectorAdd(graph, mul_4, self.b4)
        relu_4 = nn.ReLU(graph, add_4)

        mul_5 = nn.MatrixMultiply(graph, relu_4, self.W5)
        add_5 = nn.MatrixVectorAdd(graph, mul_5, self.b5)


        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, add_5, input_y)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            value_output = graph.get_output(add_5)
            # in this case, we need to find the largest number among 10 choices
            output = self.helper_get_output(value_output)
            return value_output


    def helper_get_output(self, M):
        # for each row of matrix M, we need to find the largest value position
        output = np.zeros_like(M)
        m, n = M.shape
        for x in range(m):
            max_value = -float('inf')
            max_position = 0
            for y in range(n):
                if max_value < M[x, y]:
                    max_value = M[x, y]
                    max_position = y
            output[x, max_position] = 1.0
        return output



class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.01
        # hidden layer size
        self.h1 = 400
        self.h2 = self.num_actions
        # self.h3 = 300
        # self.h4 = 300
        # set parameters for this question
        self.W1 = nn.Variable(self.state_size, self.h1)
        self.b1 = nn.Variable(1, self.h1)
        self.W2 = nn.Variable(self.h1, self.h2)
        self.b2 = nn.Variable(1, self.h2)
        # self.W3 = nn.Variable(self.h2, self.h3)
        # self.b3 = nn.Variable(1, self.h3)
        # self.W4 = nn.Variable(self.h3, self.h4)
        # self.b4 = nn.Variable(1, self.h4)
        # self.W5 = nn.Variable(self.h4, 10)
        # self.b5 = nn.Variable(1, 10)


    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "*** YOUR CODE HERE ***"
        graph = nn.Graph([self.W1, self.b1, self.W2, self.b2])
        input_x = nn.Input(graph, states)

        mul_1 = nn.MatrixMultiply(graph, input_x, self.W1)
        add_1 = nn.MatrixVectorAdd(graph, mul_1, self.b1)
        relu_1 = nn.ReLU(graph, add_1)

        mul_2 = nn.MatrixMultiply(graph, relu_1, self.W2)
        add_2 = nn.MatrixVectorAdd(graph, mul_2, self.b2)

        if Q_target is not None:
            "*** YOUR CODE HERE ***"
            # first, calculate the loss
            input_y = nn.Input(graph, Q_target)
            loss = nn.SquareLoss(graph, add_2, input_y)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            return graph.get_output(add_2)

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.1
        # hidden layer size
        self.d1 = 500
        self.d2 = 500
        self.d3 = 500
        self.d4 = 500
        # batch_size
        self.batch_size = None
        # set parameters for this question
        # f = ReLU(h * W1 + c * W2 + b1) * W3 + b2
        self.h = None
        self.W1 = nn.Variable(self.d1, self.d2)
        self.W2 = nn.Variable(self.num_chars, self.d2)
        self.b1 = nn.Variable(1, self.d2)
        
        self.W3 = nn.Variable(self.d2, self.d1)
        self.b2 = nn.Variable(1, self.d1)

        # at last, we need to chage the matrix size from batch_size x self.d1 to batch_size x 5
        # output = f * W4 + b3
        self.W4 = nn.Variable(self.d1, 5)
        self.b3 = nn.Variable(1, 5)

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use `nn.SoftmaxLoss` as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]
        "*** YOUR CODE HERE ***"
        if not self.batch_size == batch_size:
            self.batch_size = batch_size
            self.h  = nn.Variable(self.batch_size, self.d1)
            h_data = np.zeros_like(self.h.data)
            self.h.data = h_data
        graph = nn.Graph([self.W1, self.W2, self.b1, self.b2, self.W3, self.h, self.W4, self.b3])

        # use a for loop to add all char in one word to the graph
        h_node = self.h
        for c_value in xs:
            input_c = nn.Input(graph, c_value)
            h_node = self.helper_function(graph, h_node, input_c)
        # at this time, h_node is the output of the RNN
        # we need to compare this number to y
        # g(h) = h * W4 + b3
        # size:
        # h: batch_size x h1, W4: h1 x 5, b3: 1 x 5
        mul = nn.MatrixMultiply(graph, h_node, self.W4)
        add = nn.MatrixVectorAdd(graph, mul, self.b3)

        if y is not None:
            "*** YOUR CODE HERE ***"
            input_y = nn.Input(graph, y)
            loss = nn.SquareLoss(graph, add, input_y)
            return graph

        else:
            "*** YOUR CODE HERE ***"
            value_output = graph.get_output(add)
            # in this case, we need to find the largest number among 5 choices
            # output = self.helper_get_output(value_output)
            return value_output


    def helper_function(self, graph, h, c):
        # this function helps to calculate the feature f(h, c)
        # f = ReLU(h * W1 + c * W2 + b1) * W3 + b2
        # size: (h has size batch_size x h1, c has size batch_size x 47)
        # W1: h1 x h2, W2: 47 x h2, b1: 1 x h2, W3: h2 x h1, b2: 1 x h1
        mul_1 = nn.MatrixMultiply(graph, h, self.W1)
        mul_2 = nn.MatrixMultiply(graph, c, self.W2)
        add_1 = nn.Add(graph, mul_1, mul_2)
        add_2 = nn.MatrixVectorAdd(graph, add_1, self.b1)
        relu_1 = nn.ReLU(graph, add_1)

        mul_3 = nn.MatrixMultiply(graph, relu_1, self.W3)
        add_3 = nn.MatrixVectorAdd(graph, mul_3, self.b2)
        return add_3

    # def helper_get_output(self, M):
    #     # for each row of matrix M, we need to find the largest value position
    #     output = np.zeros_like(M)
    #     m, n = M.shape
    #     for x in range(m):
    #         max_value = -float('inf')
    #         max_position = 0
    #         for y in range(n):
    #             if max_value < M[x, y]:
    #                 max_value = M[x, y]
    #                 max_position = y
    #         output[x, max_position] = 1.0
    #     return output
