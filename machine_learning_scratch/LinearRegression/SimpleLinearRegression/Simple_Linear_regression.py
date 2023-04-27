import numpy as np

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = 0
        self.weight = 0

    def fit(self, X, y):
        self.weight, self.bias, cost_array = SimpleLinearRegression._gradient_descent(X, y, self.weight, self.bias, self.learning_rate, self.n_iterations)
        return self.weight, self.bias, cost_array

    def predict(self, X):
        y_predicted = np.dot(X, self.weight) + self.bias
        return y_predicted

    @staticmethod
    def _cost_function(x, y, w, b):
        n = x.shape[0]
        cost = 0

        for i in range(n):
            f_wb = w * x[i] + b
            cost += (f_wb - y[i]) ** 2
        total_cost = 1 / (2*n) * cost
        return total_cost

    @staticmethod
    def _gradient_function(x, y, w, b):

        """
        Computes the derivatives for grtadient descent
        Args:
        x (ndarray (n,)): Data, n examples 
        y (ndarray (n,)): target values
        w, b (scalar)   : model parameters 

        Returns
        dj_dw (scalar): The gradient of the cost function for the parameters w
        dj_db (scalar): The gradient of the cost function for the parameters b    
        """

        n = x.shape[0]    
        dj_dw = 0
        dj_db = 0

        for i in range(n):
            f_wb = w * x[i] + b
            dj_dw_i = (f_wb - y[i]) * x[i]
            dj_db_i = f_wb - y[i]
            dj_dw += dj_dw_i
            dj_db += dj_db_i
        dj_dw = dj_dw / n 
        dj_db = dj_db / n
        return dj_dw, dj_db

    @staticmethod
    def _gradient_descent(x, y, w_in, b_in, alpha, num_iters): 
        """
        Performs gradient descent to fit w,b. Updates w,b by taking 
        num_iters gradient steps with learning rate alpha

        Args:
        x (ndarray (n,)):    Data, n examples 
        y (ndarray (n,)):    Target values
        w_in, b_in (scalar): Initial values of model parameters  
        alpha (float):       Learning rate
        num_iters (int):     Number of iterations to run gradient descent
        cost_function:       Function to call to produce cost
        gradient_function:   Function to call to produce gradient

        Returns:
        w (scalar):          Updated value of parameter after running gradient descent
        b (scalar):          Updated value of parameter after running gradient descent
        cost_history (List): History of cost values
        p_history (list):    History of parameters [w,b] 
        """

        b = b_in
        w = w_in
        cost_history = []

        for i in range(num_iters):
            # Calculate the gradient and update the parameters using gradient_function
            dj_dw, dj_db = SimpleLinearRegression._gradient_function(x, y, w, b)     

            b = b - alpha * dj_db                            
            w = w - alpha * dj_dw                            

            # Save cost J at each iteration
            cost_history.append(SimpleLinearRegression._cost_function(x, y, w , b))

        return w, b, cost_history
