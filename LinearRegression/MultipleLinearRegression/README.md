no caso da regressão linear múltipla, em que há duas ou mais variáveis independentes, a implementação do algoritmo se torna mais complexa. É necessário aplicar a mesma lógica utilizada na regressão linear simples, mas com ajustes para lidar com mais de uma variável independente. No caso, esse ajuste contém a multiplicação de escalar por vetores. Esse problema de álgebra linear pode ser resolvido usando a biblioteca Numpy.

A função que estaremos usando para realizar predições com a regressão linear múltipla é:
$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = \mathbf{w} \cdot \mathbf{x}^{(i)} + b  \tag{4} $$
Mas agora com $\mathbf{w}$ e $\mathbf{x}^{(i)}$ sendo vetores.

Para calcular a função de custo utilizaremos a mesma função que de regressão linear simples:
$$J(\mathbf{w},b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})^2 \tag{3}$$ 
