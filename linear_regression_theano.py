
import numpy as np
import theano.tensor as T
import theano

class LinearRegression(object):
  """
  Use a straight line to fit your data & predict future values.
  The interface is similar one that's provided by scikit-learn
  """

  def __init__(self, lr=0.01, epochs=1000):
    self.lr = lr
    self.epochs = epochs
    # parameters
    self.theta = None
    # training function
    self.train_fn = None
    # predict function
    self.predict_fn = None

  def _transform_data(self, X_data):
    """Add ones as the intercept terms"""

    data_shape = np.shape(X_data)
    X_data_cpy = np.ones((data_shape[0], data_shape[1] + 1))
    X_data_cpy[:, 1:] = np.copy(X_data)

    return X_data_cpy

  def fit(self, X_data, y_target):
    """
    Fit the datasets to the model.

    This method will initialize the parameters (theta) train function,
    predict function, and cost function.
    """

    X_data_cpy = self._transform_data(X_data)

    if not self.theta:
      self.theta = theano.shared(np.random.randn(X_data_cpy.shape[1], 1), name='theta')
    else:
      self.theta.set_value(np.random.randn(X_data_cpy.shape[1], 1))

    X = T.dmatrix('X')
    y = T.dmatrix('y')

    prediction = T.dot(X, self.theta)
    self.predict_fn = theano.function([X], prediction)

    cost = (1 / (2.0 * X_data_cpy.shape[0])) * T.sum(T.pow(prediction - y, 2))
    cost_grad = T.grad(cost, wrt=self.theta)
    self.train_fn = theano.function(inputs=[X, y],
                                 outputs=[prediction, cost],
                                 updates=[(self.theta, self.theta - self.lr * cost_grad)])
    self.costs = []
    for i in range(self.epochs):
      pred, err = self.train_fn(X_data_cpy, y_target)
      self.costs.append(err)

    return self

  def predict(self, X):
    return self.predict_fn(self._transform_data(X))

  def get_params(self):
    return self.theta.get_value()
