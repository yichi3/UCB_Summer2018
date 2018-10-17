import numpy as np

def main():
    """
    This is sample code for linear regression, which demonstrates how to use the
    Graph class.

    Once you have answered Questions 2 and 3, you can run `python nn.py` to
    execute this code.
    """

    # This is our data, where x is a 4x2 matrix and y is a 4x1 matrix
    x = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
    y = np.dot(x, np.array([[7.],
                            [8.]])) + 3

    # Let's construct a simple model to approximate a function from 2D
    # points to numbers, f(x) = x_0 * m_0 + x_1 * m_1 + b
    # Here m and b are variables (trainable parameters):
    m = Variable(2,1)
    b = Variable(1)

    # We train our network using batch gradient descent on our data
    for iteration in range(10000):
        # At each iteration, we first calculate a loss that measures how
        # good our network is. The graph keeps track of all operations used
        graph = Graph([m, b])
        input_x = Input(graph, x)
        input_y = Input(graph, y)
        xm = MatrixMultiply(graph, input_x, m)
        xm_plus_b = MatrixVectorAdd(graph, xm, b)
        loss = SquareLoss(graph, xm_plus_b, input_y)
        # Then we use the graph to perform backprop and update our variables
        graph.backprop()
        graph.step(0.01)

    # After training, we should have recovered m=[[7],[8]] and b=[3]
    print("Final values are: {}".format([m.data[0,0], m.data[1,0], b.data[0]]))
    assert np.isclose(m.data[0,0], 7)
    assert np.isclose(m.data[1,0], 8)
    assert np.isclose(b.data[0], 3)
    print("Success!")

class Graph(object):
    """
    A graph that keeps track of the computations performed by a neural network
    in order to implement back-propagation.

    Each evaluation of the neural network (during both training and test-time)
    will create a new Graph. The computation will add nodes to the graph, where
    each node is either a DataNode or a FunctionNode.

    A DataNode represents a trainable parameter or an input to the computation.
    A FunctionNode represents doing a computation based on two previous nodes in
    the graph.

    The Graph is responsible for keeping track of all nodes and the order they
    are added to the graph, for computing gradients using back-propagation, and
    for performing updates to the trainable parameters.

    For an example of how the Graph can be used, see the function `main` above.
    """

    def __init__(self, variables):
        """
        Initializes a new computation graph.

        variables: a list of Variable objects that store the trainable parameters
            for the neural network.

        Hint: each Variable is also a node that needs to be added to the graph,
        so don't forget to call `self.add` on each of the variables.
        """
        "*** YOUR CODE HERE ***"
        # in this problem, we need to store the following staff:
        # 1. all nodes including variable nodes and function nodes
        self.node_list =[]
        # use a dict to store the forward and backward value
        # key is the node and value is a list that contains two values. one forward and one backward
        self.values = dict()
        # first, include all trainable variables and record the number of them
        self.num_trainable = len(variables)
        for v in variables:
            self.add(v)
        # 2. calculate the value of all variable node.
        # this process is done in seld.add function
        # 3. do the back propagation.
        # this process is done in self.backprop function
        # use a dict to store the forward and backward value
        # key is the node and value is a list that contains two values. one forward and one backward
        # self.alpha = 0.0


    def get_nodes(self):
        """
        Returns a list of all nodes that have been added to this Graph, in the
        order they were added. This list should include all of the Variable
        nodes that were passed to `Graph.__init__`.

        Returns: a list of nodes
        """
        "*** YOUR CODE HERE ***"
        return self.node_list

    def get_inputs(self, node):
        """
        Retrieves the inputs to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a list of numpy arrays

        Hint: every node has a `.get_parents()` method
        """
        "*** YOUR CODE HERE ***"
        # apart from the variable nodes, input nodes have no parent (self.parent is empty list)
        # we can recursively call this function to get the inputs to this node
        # special case, if the node is a variable node
        if node in self.node_list[0:self.num_trainable]:
            return []
        return self.helper_get_inputs(node)
        
    def helper_get_inputs(self, node):
        inputs_list = []
        if node.get_parents() == []:
            return [node.forward([])]
        for parent in node.get_parents():
            inputs = self.helper_get_inputs(parent)
            inputs_list.extend(inputs)
        return inputs_list


    def get_output(self, node):
        """
        Retrieves the output to a node in the graph. Assume the `node` has
        already been added to the graph.

        Returns: a numpy array or a scalar
        """
        "*** YOUR CODE HERE ***"        
        # return the self.values[0]
        return self.values[node][0]


    def get_gradient(self, node):
        """
        Retrieves the gradient for a node in the graph. Assume the `node` has
        already been added to the graph.

        If `Graph.backprop` has already been called, this should return the
        gradient of the loss with respect to the output of the node. If
        `Graph.backprop` has not been called, it should instead return a numpy
        array with correct shape to hold the gradient, but with all entries set
        to zero.

        Returns: a numpy array
        """
        "*** YOUR CODE HERE ***"
        return self.values[node][1]

    def add(self, node):
        """
        Adds a node to the graph.

        This method should calculate and remember the output of the node in the
        forwards pass (which can later be retrieved by calling `get_output`)
        We compute the output here because we only want to compute it once,
        whereas we may wish to call `get_output` multiple times.

        Additionally, this method should initialize an all-zero gradient
        accumulator for the node, with correct shape.
        """
        "*** YOUR CODE HERE ***"
        # first, we need to add the node to the node list
        self.node_list.append(node)
        # calculate the forward value here
        parents_list = node.get_parents()
        # if this node is a data node
        if parents_list == []:
            forward_value = node.forward(parents_list)
            # forward_value = node.data
            backward_value = np.zeros_like(forward_value)
            self.values[node] = [forward_value, backward_value]
        # if this node is a function node
        else:
            parents_value = [self.values[p][0] for p in parents_list]
            forward_value = node.forward(parents_value)
            backward_value = np.zeros_like(forward_value)
            self.values[node] = [forward_value, backward_value]



    def backprop(self):
        """
        Runs back-propagation. Assume that the very last node added to the graph
        represents the loss.

        After back-propagation completes, `get_gradient(node)` should return the
        gradient of the loss with respect to the `node`.

        Hint: the gradient of the loss with respect to itself is 1.0, and
        back-propagation should process nodes in the exact opposite of the order
        in which they were added to the graph.
        """
        loss_node = self.get_nodes()[-1]
        assert np.asarray(self.get_output(loss_node)).ndim == 0

        "*** YOUR CODE HERE ***"
        # first, set the gradient of loss to be 1.0
        # since loss is just a scalar, the gradient is actually 1.0, not an numpy array
        self.values[loss_node][1] = 1.0
        # do recursion for the whole graph to set the gradient for all node
        self.helper_backprop(loss_node)

    def helper_backprop(self, current_node):
        parents_list = current_node.get_parents()
        # base case. If the current node is a data node
        if parents_list == []:
            return
        parents_value = [self.values[p][0] for p in parents_list]
        gradient = self.values[current_node][1]
        parents_gradient = current_node.backward(parents_value, gradient)
        for index in range(len(parents_list)):
            node = parents_list[index]
            self.values[node][1] += parents_gradient[index]
        # go to the top level of current node
        if len(parents_list) == 1:
            self.helper_backprop(parents_list[0])
        elif parents_list[0] == parents_list[1]:
            self.helper_backprop(parents_list[0])
        else:
            for node in parents_list:
                self.helper_backprop(node)


    def step(self, step_size):
        """
        Updates the values of all variables based on computed gradients.
        Assume that `backprop()` has already been called, and that gradients
        have already been computed.

        Hint: each Variable has a `.data` attribute
        """
        "*** YOUR CODE HERE ***"
        # self.alpha = step_size
        # what we will do in this function is to update all trainable variables
        # w <- w + self.alpha * gradient
        for node in self.node_list[0:self.num_trainable]:
            node.data -= step_size * self.values[node][1]


class DataNode(object):
    """
    DataNode is the parent class for Variable and Input nodes.

    Each DataNode must define a `.data` attribute, which represents the data
    stored at the node.
    """

    @staticmethod
    def get_parents():
        # A DataNode has no parent nodes, only a `.data` attribute
        return []

    def forward(self, inputs):
        # The forwards pass for a data node simply returns its data
        return self.data

    @staticmethod
    def backward(inputs, gradient):
        # A DataNode has no parents or inputs, so there are no gradients to
        # compute in the backwards pass
        return []

class Variable(DataNode):
    """
    A Variable stores parameters used in a neural network.

    Variables should be created once and then passed to all future Graph
    constructors. Use `.data` to access or modify the numpy array of parameters.
    """

    def __init__(self, *shape):
        """
        Initializes a Variable with a given shape.

        For example, Variable(5) will create 5-dimensional vector variable,
        while Variable(10, 10) will create a 10x10 matrix variable.

        The initial value of the variable before training starts can have a big
        effect on how long the network takes to train. The provided initializer
        works well across a wide range of applications.
        """
        assert shape
        limit = np.sqrt(3.0 / np.mean(shape))
        self.data = np.random.uniform(low=-limit, high=limit, size=shape)

class Input(DataNode):
    """
    An Input node packages a numpy array into a node in a computation graph.
    Use this node for inputs to your neural network.

    For trainable parameters, use Variable instead.
    """

    def __init__(self, graph, data):
        """
        Initializes a new Input and adds it to a graph.
        """
        assert isinstance(data, np.ndarray), "data must be a numpy array"
        assert data.dtype.kind == "f", "data must have floating-point entries"
        self.data = data
        graph.add(self)

class FunctionNode(object):
    """
    A FunctionNode represents a value that is computed based on other nodes in
    the graph. Each function must implement both a forward and backward pass.
    """

    def __init__(self, graph, *parents):
        self.parents = parents
        graph.add(self)

    def get_parents(self):
        return self.parents

    @staticmethod
    def forward(inputs):
        raise NotImplementedError

    @staticmethod
    def backward(inputs, gradient):
        raise NotImplementedError

class Add(FunctionNode):
    """
    Adds two vectors or matrices, element-wise

    Inputs: [x, y]
        x may represent either a vector or a matrix
        y must have the same shape as x
    Output: x + y
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        x, y = inputs
        return x + y

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # return a list of backward result
        # e.g. if f = f(C) and C = A + B
        # then, dC/dA = I = dC/dB
        # therefore, df/dA = df/dC = gradient
        return [gradient, gradient]



class MatrixMultiply(FunctionNode):
    """
    Represents matrix multiplication.

    Inputs: [A, B]
        A represents a matrix of shape (n x m)
        B represents a matrix of shape (m x k)
    Output: a matrix of shape (n x k)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        A, B = inputs
        return np.dot(A, B)

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # C = AB
        # df/dA = C' * B^T
        # df/dB = A^T * C'
        A, B = inputs
        return [np.dot(gradient, B.T), np.dot(A.T, gradient)]

class MatrixVectorAdd(FunctionNode):
    """
    Adds a vector to each row of a matrix.

    Inputs: [A, x]
        A represents a matrix of shape (n x m)
        x represents a vector (m)
    Output: a matrix of shape (n x m)
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        A, x = inputs
        return A + x

    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        A, x = inputs
        # actually, A + x = A + np.ones((n, 1)) * x
        # therefore, df/dA = gradient
        # df/dx = sum gradient for each column
        dA = gradient
        dx = np.sum(gradient, axis=0)
        return [dA, dx]



class ReLU(FunctionNode):
    """
    An element-wise Rectified Linear Unit nonlinearity: max(x, 0).
    This nonlinearity replaces all negative entries in its input with zeros.

    Input: [x]
        x represents either a vector or matrix
    Output: same shape as x, with no negative entries
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        x = inputs[0]
        # y = np.zeros(x.shape)
        # z = np.maximum(x, y)
        z = np.where(x >= 0, x, 0)
        return z



    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # algorithm for this part
        # for all element in inputs, if this element is non-negative
        # the gradient of the element in the same position is preserved
        # otherwise, the element in gradient is converted to 0
        x = inputs[0]
        z = np.where(x >= 0, gradient, 0)
        return [z]




class SquareLoss(FunctionNode):
    """
    Inputs: [a, b]
        a represents a matrix of size (batch_size x dim)
        b must have the same shape as a
    Output: a number

    This node first computes 0.5 * (a[i,j] - b[i,j])**2 at all positions (i,j)
    in the inputs, which creates a (batch_size x dim) matrix. It then calculates
    and returns the mean of all elements in this matrix.
    """

    @staticmethod
    def forward(inputs):
        "*** YOUR CODE HERE ***"
        a, b = inputs
        diff = a - b
        square_loss = 0.5 * diff * diff
        m = np.mean(square_loss)
        return m


    @staticmethod
    def backward(inputs, gradient):
        "*** YOUR CODE HERE ***"
        # algorithm:
        # let d = 0.5 * (a-b) * (a-b), where d is a matrix with the same shape as a and b
        # f = f(y) where y is given as following:
        # y = mean value of d = sum(d_ij) / N, where N is the number of total elements in d
        # therefore d_ij = 0.5*(a_ij-b_ij)^2
        # dy/d(a_ij) = d(sum(d_ij)/N) / d(a_ij)
        #            = d(d_ij) / d(a_ij) / N   (we can get rid of the sum here because the only thing
        #                                       in the sum that depends on a_ij is d_ij term)
        #            = d(0.5(a_ij-b_ij)^2) / d(a_ij) / N
        #            = (a_ij - b_ij) / N
        # df/d(a_ij) = df/dy * dy/d(a_ij) = gradient * dy/d(a_ij)
        # similar for b_ij
        # dy/d(b_ij) = (b_ij - a_ij) / N
        a, b = inputs
        m, n = a.shape
        N = m * n
        diff_ab = gradient * (a - b) / N
        diff_ba = gradient * (b - a) / N
        return [diff_ab, diff_ba]



class SoftmaxLoss(FunctionNode):
    """
    A batched softmax loss, used for classification problems.

    IMPORTANT: do not swap the order of the inputs to this node!

    Inputs: [logits, labels]
        logits: a (batch_size x num_classes) matrix of scores, that is typically
            calculated based on previous layers. Each score can be an arbitrary
            real number.
        labels: a (batch_size x num_classes) matrix that encodes the correct
            labels for the examples. All entries must be non-negative and the
            sum of values along each row should be 1.
    Output: a number

    We have provided the complete implementation for your convenience.
    """
    @staticmethod
    def softmax(input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def forward(inputs):
        softmax = SoftmaxLoss.softmax(inputs[0])
        labels = inputs[1]
        assert np.all(labels >= 0), \
            "Labels input to SoftmaxLoss must be non-negative. (Did you pass the inputs in the right order?)"
        assert np.allclose(np.sum(labels, axis=1), np.ones(labels.shape[0])), \
            "Labels input to SoftmaxLoss do not sum to 1 along each row. (Did you pass the inputs in the right order?)"

        return np.mean(-np.sum(labels * np.log(softmax), axis=1))

    @staticmethod
    def backward(inputs, gradient):
        softmax = SoftmaxLoss.softmax(inputs[0])
        return [
            gradient * (softmax - inputs[1]) / inputs[0].shape[0],
            gradient * (-np.log(softmax)) / inputs[0].shape[0]
        ]

if __name__ == '__main__':
    main()
