import numpy as np
import copy

class MultipleLinearRegression:
    def __init__(self, n_features, learning_rate=0.001, n_iterations=1000):
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
        cost_function
        Argumentos:
        X (ndarray (m,n)): Dados, m amostras com n features
        y (ndarray (m,)) : Valores target
        w (ndarray (n,)) : Parâmetros do modelo 
        b (escalar)      : Parâmetro do modelo

        Retorna:
        cost (escalar)
        """

        m = X.shape[0]
        cost = 0.0
        for i in range(m):                                
            f_wb_i = np.dot(X[i], w) + b           #(n,).(n,) = escalar
            cost = cost + (f_wb_i - y[i])**2       #escalar
        cost = cost / (2 * m)                      #escalar
        return cost

    @staticmethod
    def _gradient_function(X, y, w, b): 
        """
        Formula as derivadas dos parâmetros w e b
        Argumentos:
        X (ndarray (m,n)): Dados, m amostras com n features
        y (ndarray (m,)) : Valores target
        w (ndarray (n,)) : Parâmetros do modelo 
        b (escalar)      : Parâmetros do modelo

        Retorna:
        dj_dw (ndarray (n,)): Todas as derivadas de todos os parâmetros w. 
        dj_db (escalar)     : A derivada para o parâmetro b. 
        """

        m, n = X.shape
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
        Aplica a descida de gradiente para computar os valores w e b. Atualiza os valores de w e b usando o
        número de iterações do gradiente junto com a taxa de aprendizado alpha.

        Argumentos:
        X (ndarray (m,n))   : Dados, m amostras com n features
        y (ndarray (m,))    : Valores target
        w_in (ndarray (n,)) : Valores w iniciais
        b_in (escalar)      : Valor b inicial
        alpha (float)       : Taxa de aprendizado
        num_iters (int)     : Número de iterações para se atualizar os parâmetros w e b

        Retorna:
        w (ndarray (n,))   : Valores atualizados para os parâmetros w
        b (escalar)        : Valor atualizado para o parâmetro b
        cost_history (List): Histórico dos valores da função de custo
        """

        # Um array para armazenar o histórico da função de custo J e outro para armazenar o w de cada iteração
        cost_history = []
        w = copy.deepcopy(w_in)
        b = b_in

        for i in range(num_iters):

            # Calcula as derivadas
            dj_db, dj_dw = MultipleLinearRegression._gradient_function(X, y, w, b)

            # Atualiza o w e b subtraindo o w e b antigos pela taxa de aprendizado e depois multiplando pelas suas respectivas derivadas
            w = w - alpha * dj_dw
            b = b - alpha * dj_db

            # Armazena no array de histórico da função de custo
            cost_history.append(MultipleLinearRegression._cost_function(X, y, w, b))

        return w, b, cost_history
