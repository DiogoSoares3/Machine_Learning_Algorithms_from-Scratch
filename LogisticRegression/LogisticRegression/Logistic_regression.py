import numpy as np
import copy

class LogisticRegression:
    def __init__(self, n_features, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = 0
        self.weights = np.zeros((n_features,))

    def fit(self, X, y):
        self.weights, self.bias, cost_array = LogisticRegression._gradient_descent(X, y, self.weights, self.bias, self.learning_rate, self.n_iterations)
        return self.weights, self.bias, cost_array

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_predicted = LogisticRegression._sigmoid(z)
        y_predicted_comp = [1 if i >= 0.5 else 0 for i in y_predicted]
        return y_predicted_comp
    
    @staticmethod
    def _sigmoid(z):
        a = 1/(1+np.exp(-z))
        a = np.clip(a, 1e-10, 1 - 1e-10)           # Evita overflow
        return a
    
    @staticmethod
    def _cost_function(X, y, w, b):
        """
        Função de custo

        Args:
        X (ndarray (m,n)): Dados, m amostras com n features
        y (ndarray (m,)) : Valores target
        w (ndarray (n,)) : Parâmetros do modelo  
        b (escalar)      : Parâmetro do modelo
        
        Retorna:
        cost (escalar)
        """

        m, n = X.shape
        cost = 0.0
        for i in range(m):
            z_i = np.dot(X[i],w) + b                                             #(n,)(n,) ou (n,) ()
            f_wb_i = LogisticRegression._sigmoid(z_i)                                                   #(n,)
            cost  += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)       # escalar
        cost = cost / m

        return cost
    
    @staticmethod
    def _gradient_function(X, y, w, b): 
        """
        Formula as derivadas dos parâmetros w e b 

        Argumentos:
        X (ndarray (m,n): Dados, m amostras com n features
        y (ndarray (m,)): Valores target
        w (ndarray (n,)): Parâmetros do modelo  
        b (escalar)     : Parâmetro do modelo
        
        Retorna:
        dj_dw (ndarray (n,)): Todas as derivadas de todos os parâmetros w.
        dj_db (escalar)     : A derivada para o parâmetro b. 
        """

        m, n = X.shape
        dj_dw = np.zeros((n,))                           #(n,)
        dj_db = 0.

        for i in range(m):
            f_wb_i = LogisticRegression._sigmoid(np.dot(X[i], w) + b)          #(n,).(n,)=escalar
            err_i  = f_wb_i  - y[i]                       #escalar
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #escalar
            dj_db = dj_db + err_i
        dj_dw = dj_dw / m                                   #(n,)
        dj_db = dj_db / m                                   #escalar
            
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
        cost_history (lista): Histórico dos valores da função de custo
        """

        # Um array para armazenar o histórico da função de custo J e outro para armazenar o w de cada iteração
        cost_history = []
        w = copy.deepcopy(w_in)  #avoid modifying global w within function
        b = b_in
        
        for i in range(num_iters):
            # Calcula as derivadas
            dj_db, dj_dw = LogisticRegression._gradient_function(X, y, w, b)   

            # Atualiza o w e b subtraindo o w e b antigos pela taxa de aprendizado e depois multiplando pelas suas respectivas derivadas
            w = w - alpha * dj_dw               
            b = b - alpha * dj_db             

            # Armazena no array de histórico da função de custo
            cost_history.append(LogisticRegression._cost_function(X, y, w, b))
            
        return w, b, cost_history
