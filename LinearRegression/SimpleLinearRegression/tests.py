import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from Simple_Linear_regression import SimpleLinearRegression

X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=645) 

w_init = 0
b_init = 0
iterations = 2000
alpha = 1.0e-2

model = SimpleLinearRegression(learning_rate=alpha, n_iterations=iterations)
w_final, b_final, cost_array = model.fit(X, y)
print(f"(w,b) calculados pela descida de gradiente: ({w_final[0]:.4f},{b_final[0]:.4f})")
print(f"Valor mínimo da função de custo: {cost_array[-1][0]:.4f}")
print(f"Convertendo para RMSE (mesma unidade de medida do y): {np.sqrt(cost_array[-1][0]*2)}")

y_pred_line = model.predict(X)
print(y_pred_line)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='b', marker='o', s=40)
plt.plot(X, y_pred_line, c='r')
plt.show()