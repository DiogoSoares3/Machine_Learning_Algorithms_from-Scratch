import numpy as np
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Logistic_regression import LogisticRegression

bc = load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=999)

iterations = 1000
alpha = 1.0e-5

model = LogisticRegression(n_features=X.shape[1], learning_rate=alpha, n_iterations=iterations)
w_final, b_final, cost_array = model.fit(X_train, y_train)
print(f"w calculados pela descida de gradiente: ({w_final})")
print(f"b calculado pela descida de gradiente: ({b_final})")
print(f"Valor mínimo da função de custo: {cost_array}")

y_pred = model.predict(X_test)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

print(f"Accuracy: {accuracy(y_test, y_pred)}")

