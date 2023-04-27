import numpy as np
import copy

class MultipleLinearRegression:
    def __init__(self, n_features=1 , learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = 0
        self.weights = np.zeros((n_features,))
    
    
    def fit(self, X, y):
        self.weights, self.bias, cost_array = MultipleLinearRegression._gradient_descent(X, y, self.weights, self.bias, self.learning_rate, self.n_iterations)
        return self.weights, self.bias, cost_array

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
    
    @staticmethod
    def _cost_function(X, y, w, b): 
        """
        compute cost
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        cost (scalar): cost
        """
        m = X.shape[0]
        cost = 0.0
        for i in range(m):                                
            f_wb_i = np.dot(X[i], w) + b           #(n,).(n,) = scalar
            cost = cost + (f_wb_i - y[i])**2       #scalar
        cost = cost / (2 * m)                      #scalar
        return cost
    
    @staticmethod
    def _gradient_function(X, y, w, b): 
        """
        Computes the gradient for linear regression 
        Args:
        X (ndarray (m,n)): Data, m examples with n features
        y (ndarray (m,)) : target values
        w (ndarray (n,)) : model parameters  
        b (scalar)       : model parameter
        
        Returns:
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar):       The gradient of the cost w.r.t. the parameter b. 
        """

        m, n = X.shape           #(number of examples, number of features)
        dj_dw = np.zeros((n,))
        dj_db = .0

        for i in range(m):                             
            err = (np.dot(X[i], w) + b) - y[i]
            for j in range(n):                         
                dj_dw[j] = dj_dw[j] + err * X[i, j]    
            dj_db = dj_db + err                        
        dj_dw = dj_dw / m                                
        dj_db = dj_db / m                                

        return dj_db, dj_dw
    
    @staticmethod
    def _gradient_descent(X, y, w_in, b_in, alpha, num_iters): 
        """
        Performs batch gradient descent to learn w and b. Updates w and b by taking 
        num_iters gradient steps with learning rate alpha
        
        Args:
        X (ndarray (m,n))   : Data, m examples with n features
        y (ndarray (m,))    : target values
        w_in (ndarray (n,)) : initial model parameters  
        b_in (scalar)       : initial model parameter
        cost_function       : function to compute cost
        gradient_function   : function to compute the gradient
        alpha (float)       : Learning rate
        num_iters (int)     : number of iterations to run gradient descent
        
        Returns:
        w (ndarray (n,)) : Updated values of parameters 
        b (scalar)       : Updated value of parameter 
        """

        # An array to store cost J and w's at each iteration primarily for graphing later
        cost_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):

            # Calculate the gradient and update the parameters
            dj_db, dj_dw = MultipleLinearRegression._gradient_function(X, y, w, b)

            # Update Parameters using w, b, alpha and gradient
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            cost_history.append(MultipleLinearRegression._cost_function(X, y, w, b))

        return w, b, cost_history