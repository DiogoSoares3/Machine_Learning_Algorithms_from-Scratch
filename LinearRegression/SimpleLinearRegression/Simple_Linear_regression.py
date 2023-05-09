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
        Formula as derivadas dos parâmetros w e b
        Argumentos:
        x (ndarray (n,)): Dados, n amostras 
        y (ndarray (n,)): Valores target
        w, b (escalar)  : Parâmetros do modelo

        Retorna
        dj_dw (escalar): Derivada do parâmetro w do modelo
        dj_db (escalar): Derivada do parâmetro b do modelo
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
        Aplica a descida de gradiente para computar os valores w e b. Atualiza os valores de w e b usando o
        número de iterações do gradiente junto com a taxa de aprendizado alpha.

        Args:
        x (ndarray (n,)):     Dados, n amostras 
        y (ndarray (n,)):     Valores target
        w_in, b_in (escalar): Valores iniciais para os valores w e b
        alpha (float):        Taxa de aprendizado
        num_iters (int):      Número de iterações para se atualizar os parâmetros w e b

        Retorna:
        w (escalar)        : Valor atualizado para o parâmetro w
        b (escalar)        : Valor atualizado para o parâmetro b
        cost_history (List): Histórico dos valores da função de custo
        """

        b = b_in
        w = w_in

        # Um array para armazenar o histórico da função de custo J
        cost_history = []

        for i in range(num_iters):
            # Calcula as derivadas
            dj_dw, dj_db = SimpleLinearRegression._gradient_function(x, y, w, b)     

            # Atualiza o w e b utilizando a taxa de aprendizado, o w e b antigos e suas respectivas derivadas
            b = b - alpha * dj_db                            
            w = w - alpha * dj_dw                            

            # Salva a função de custo
            cost_history.append(SimpleLinearRegression._cost_function(x, y, w , b))

        return w, b, cost_history
