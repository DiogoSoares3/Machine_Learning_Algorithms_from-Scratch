# Regressao Linear Simples

No caso da regressão linear simples, em que há apenas uma variável independente, o algoritmo se torna relativamente simples. Primeiramente, é necessário calcular a média e o desvio padrão dos dados. Em seguida, é aplicada a descida de gradiente para encontrar os valores ideais dos coeficientes da equação linear. Por fim, é possível utilizar a equação para fazer previsões.

A função que usamos para realizar predições com a regressão linear simples é:
$$f_{w,b}(x^{(i)}) = wx^{(i)} + b$$

Para a função de custo J estaremos usando:
$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Para a atualização dos parâmetros w e b da regressão:

$$\begin{align*} \text{repetir}&\text{ até convergir:} \ \lbrace \newline
\  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \  \ \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$

Com o cálculo das derivadas sendo representadas por:

$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \\\
\end{align}
\text{m = n° amostras (linhas)} \
$$

Com isso nós podemos realizar a descidade de gradiente e achar os valores ideais para os parâmetros da regressão.
