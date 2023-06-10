# Bike Sharing Demand Prediction

Este projeto tem como objetivo criar um modelo de previsão de demanda de compartilhamento de bicicletas com base em diferentes variáveis, como temperatura, umidade, velocidade do vento, entre outras.

## Descrição do Conjunto de Dados

O conjunto de dados utilizado neste projeto é chamado de "bike-sharing-daily.csv". Ele contém informações diárias sobre o compartilhamento de bicicletas, incluindo variáveis como temporada, ano, mês, feriado, dia da semana, condições climáticas, temperatura, umidade, velocidade do vento e contagem total de bicicletas alugadas.

## Pré-processamento dos Dados

Antes de treinar o modelo, é realizado um pré-processamento nos dados. Isso inclui a remoção de colunas irrelevantes, como "instant", "casual" e "registered". Além disso, algumas variáveis categóricas são convertidas em variáveis dummy para facilitar o uso em um modelo de aprendizado de máquina. A normalização dos valores de saída (coluna "cnt") também é realizada usando a técnica de escalonamento MinMax.

## Construção e Treinamento do Modelo

O modelo de previsão é construído usando a biblioteca TensorFlow. É criada uma rede neural com camadas densas, utilizando ativações ReLU. O modelo é compilado com o otimizador Adam e a função de perda "mean_squared_error". Em seguida, é realizado o treinamento do modelo utilizando os dados de treinamento, com um número de epochs definido.

## Avaliação do Modelo

Após o treinamento, o modelo é avaliado usando os dados de teste. São calculadas várias métricas de avaliação, incluindo Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R2 Square e Adjusted R2. Essas métricas fornecem uma medida da precisão do modelo na previsão da demanda de compartilhamento de bicicletas.

## Resultados e Visualizações

Os resultados da avaliação do modelo são apresentados no console. Além disso, é gerado um gráfico mostrando o progresso da função de perda durante o treinamento.

## Requisitos de Instalação

Para executar este projeto, você precisará ter as seguintes bibliotecas instaladas:

- TensorFlow
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

Certifique-se de ter essas bibliotecas instaladas antes de executar o código.

## Como Executar o Projeto

1. Faça o download do arquivo "bike-sharing-daily.csv" e certifique-se de que ele está no mesmo diretório do arquivo Python.
2. Instale as bibliotecas necessárias listadas nos requisitos de instalação.
3. Execute o arquivo Python "bike-sharing-prediction.py".
4. Os resultados da avaliação do modelo serão exibidos no console e o gráfico da função de perda será gerado.

Sinta-se à vontade para ajustar o código ou explorar diferentes configurações do modelo de acordo com suas necessidades e objetivos específicos.
