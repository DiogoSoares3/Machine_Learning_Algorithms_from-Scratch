# Regressão Linear do zero (Scratch)

Para a implementação do algoritmo, foram utilizados métodos como a função de custo e a descida de gradiente. A função de custo é utilizada para medir a diferença entre as previsões do modelo e os valores reais dos dados. Já a descida de gradiente é um método iterativo utilizado para minimizar a função de custo, ajustando os parâmetros do modelo a cada iteração.

No caso da regressão linear simples, em que há apenas uma variável independente, o algoritmo se torna relativamente simples. Primeiramente, é necessário calcular a média e o desvio padrão dos dados. Em seguida, é aplicada a descida de gradiente para encontrar os valores ótimos dos coeficientes da equação linear. Por fim, é possível utilizar a equação para fazer previsões.

Já no caso da regressão linear múltipla, em que há duas ou mais variáveis independentes, a implementação do algoritmo se torna mais complexa. É necessário aplicar a mesma lógica utilizada na regressão linear simples, mas com ajustes para lidar com mais de uma variável independente. No caso, esse ajuste contém a multiplicação de escalar por vetores. Esse problema de álgebra linear pode ser resolvido usando a biblioteca Numpy.

Em ambos os casos, a construção do algoritmo permite uma maior compreensão do funcionamento da regressão linear e dos métodos utilizados para encontrar a melhor linha de ajuste aos dados. Além disso, é uma oportunidade para praticar habilidades de programação e matemática, bem como para aprimorar a capacidade de análise de dados.
