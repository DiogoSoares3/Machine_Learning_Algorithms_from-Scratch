# Regressão Logística

Para a predição de valores, estaremos usando a função sigmoide. Ela é uma função matemática que transforma valores em uma escala contínua entre 0 e 1, o que a torna útil na regressão logística. Na regressão logística, a função sigmoide é usada para modelar a probabilidade de um evento binário ocorrer

A função sigmoide g é representada pela equação:

$$g(z^{(i)}) \frac{1}{1+e^{-z^{(i)}}}$$
                  com
$$z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\\$$

Para regressão logísitca, a fórmula para a função de custo J é:

$$ J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right]$$

Onde
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ é o custo de um único ponto de dados (em uma única linha), que é:

    $$loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right)$$
    
*  onde m é o número de amostras (linhas) no dataset e

$$
\begin{align}
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\\
  z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\\
  g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}
\end{align}
$$

Para a descida de gradiente, temos a mesma fórmula que a da regressão linear, mas agora com $$f_{\mathbf{w},b}(\mathbf{x^{(i)}})$$ sendo a função sigmoide

$$\begin{align*}
&\text{repetir até convergir} \ \lbrace \\
&  w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \;\\ 
&  b = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b} \\
&\rbrace
\end{align*}$$

Onde cada iteração realiza atualizações simultâneas nos parâmetros $w_j$ para todo $j$, onde
$$\begin{align*}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{2} \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{3} 
\end{align*}$$
  
* $f_{\mathbf{w},b}(x^{(i)})$ é a predição do modelo, enquanto $y^{(i)}$ é o target.
