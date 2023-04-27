import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from Multiple_Linear_Regression import MultipleLinearRegression

n_features = 4
X, y = make_regression(n_samples=100, n_features=n_features, noise=15, random_state=32) 

w_init = 0
b_init = 0
iterations = 1000
alpha = 1.0e-2


model = MultipleLinearRegression(n_features=n_features, learning_rate=alpha, n_iterations=iterations)
w_final, b_final, cost_array = model.fit(X, y)
print(f"(w,b) calculados pela descida de gradiente: ({w_final[0]:.4f},{b_final:.4f})")
print(f"Valor mínimo da função de custo: {cost_array[-1]:.4f}")
print(f"Convertendo para RMSE (mesma unidade de medida do y): {np.sqrt(cost_array[-1]*2)}")

y_pred_line = model.predict(X)
print(y_pred_line)

fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(cost_array)
ax2.plot(200 + np.arange(len(cost_array[200:])), cost_array[200:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()