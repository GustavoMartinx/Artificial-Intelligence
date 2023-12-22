# Artificial Intelligence

This repository contains some projects developed during the Artificial Intelligence discipline of the Bachelor of Computer Science course at the Federal Technological University of Paraná.

All projects were implemented in Python and below are general descriptions about each one of them, as well as information on how to execute them.

<br>

## Final Project - Simpsons Recognition

We chose as the theme of our final project the implementation of an intelligent system that performs the recognition of the main characters of the animated series "The Simpsons" in images using Machine Learning techniques.

To this end, a Convolutional Neural Network (CNN) was used to perform feature extraction, as well as different classifiers for evaluation of the dataset, which were: K-Nearest Neighbors, Support Vector Machine (SVM), Multilayer Perceptron (MLP) and Classifier Combination.

From these methods, we achieved good performance with an accuracy rate (F1 score) of 74% for the artificial neural network model (the MLP) and 79% for the SVM with the use of the data augmentation technique.

In short, the recognition of the characters is not a trivial task. From this perspective, even though the use of more data for training can significantly increase accuracy, it unfortunately brings with it a significant increase in computational cost. In case of interest in this work, it is recommended to read its respective article in full:

[Intelligent System for Recognition of Characters of the Animated Series "The Simpsons"](./SimpsonsRecognition/Artigo_Simpsons.pdf)


### How to Run

1. Clone this repository

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Navigate to the project directory, namely `SimpsonsRecognition/`

        cd SimpsonsRecognition/

3. Unzip the `dataset_treated.zip` dataset;

<br>

4. Ensure you have Python installed. Then install the required libraries using the following command:

        pip install -r requirements.txt

5. To extract features through the CNN that will be used by the classifiers, run:

        python feature_extraction.py

6. Now, using the extracted features, train and test the desired model:

        python classification.py

7. Additionally, it is also possible to perform a test through the combination of classifiers if you want, for that:

        python3 combine_classifiers.py

<br>


## K-Nearest Neighbors (KNN) Classifier

### Overview

This project implements the K-Nearest Neighbors (KNN) algorithm for supervised learning classification. KNN uses distance to make predictions about data clustering, assuming that similar samples are close to each other. The choice of ``k`` (number of neighbors) and the distance metric, such as Euclidean distance, significantly impact the algorithm's performance.

### Implementation

The implementation is done in Python and utilizes the Numpy, Pandas, SciKit-Learn, and Collections libraries. Key functions of the developed algorithm include ``fit`` for storing training data and ``predict`` for classification. Different values of ``k`` are explored to understand their impact on accuracy.

### Results

The article evaluates the algorithm for different ``k`` values and percentages of training data. Accuracy results for various scenarios are visualized in graphs, showing the impact of ``k`` and dataset size on classification performance, as well as offering insights into performance under different conditions.

Additionally, the report emphasizes the importance of finding a balance between the number of features and ``k`` values for optimal accuracy.

For more information on the various tests conducted, please refer to the technical report:

[KNN Report](./KNN/Report-KNN.pdf)


### How to Run

1. Clone this repository:

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Navigate to the project directory:

        cd KNN/

3. Install dependencies:

        pip install -r requirements.txt

4. Run the KNN

        python knn.py

<br>


## Classification Using K-Means Centroids as Training Set for KNN

This project aimed to use centroids generated by the _K-Means_ algorithm as a training set for the _KNN_ classifier.

### Methods and Implementation

With this goal in mind, 1000 training and test samples were extracted, each containing 132 features of digits from 0 to 9. The samples were normalized and subjected to dimensionality reduction using Principal Component Analysis (PCA), resulting in two dimensions.

Next, a variable number of centroids were generated through K-Means to be the training set for KNN, configured with ``k = 5``.

### Results and Conclusions

However, the obtained results had low accuracy, possibly due to dimensionality reduction and potential errors in the conversion between training and test sets.

To better understand the possible reasons for the imprecise results and other details of this work, check the technical report:

[K-means Report](./K-means/K-Means_Report.pdf)

### How to Run

1. Clone this repository:

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Navigate to the project directory:

        cd K-means/

3. Install dependencies:

        pip install -r requirements.txt

4. Run K-means:

        python kmeans.py



<br>
<br>
<br>



<h1> Inteligência Artificial </h1>

Este repositório contém alguns projetos desenvolvidos durante a disciplina de Inteligência Artificial do curso de Bacharelado em Ciência da Computação da Universidade Tecnológica Federal do Paraná.

Todos os projetos foram implementados em _Python_ e a seguir encontram-se descrições gerais a respeito de cada um deles, bem como informações sobre como executá-los.

<br>

## Projeto Final - Reconhecimento dos Simpsons
Escolhemos como tema do nosso projeto final a implementação de um sistema inteligente que efetua o reconhecimento das personagens principais da animação "Os _Simpsons_" em imagens por meio de técnicas de Aprendizagem de Máquina.

Para tal, utilizou-se uma Rede Neural Convolocional (CNN) a fim de realizar a extração de características, além de diferentes classificadores para avaliação do conjunto de dados, foram eles: K-vizinhos mais próximos, Máquina de Vetores de Suporte (SVM), Perceptron Multicamadas (MLP) e Combinação de Classificadores.

A partir desses métodos, atingimos bom desempenho com taxa de acerto (F1 score) de 74% para o modelo de redes neurais artificiais (o MLP) e 79% para o SVM com a utilização da técnica de _data augmentation_.

Em suma, o reconhecimento das personagens não é uma tarefa trivial. Sob essa ótica, mesmo que o uso de mais dados para treinamento possa aumentar consideravelmente a precisão, traz consigo, infelizmente, um aumento significativo do custo computacional. Em caso de interesse sobre tal trabalho, recomenda-se a leitura de seu respectivo artigo na íntegra:

[Sistema Inteligente para Reconhecimento das Personagens da Animação "Os Simpsons"](./SimpsonsRecognition/Artigo_Simpsons.pdf)

<br>


### Como Executar

1. Clone este repositório

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Acesse o diretório do projeto, a saber `SimpsonsRecognition/`

        cd SimpsonsRecognition/


3. Descompacte o conjunto de dados `dataset_treated.zip`;

<br>

4. Certifique-se de ter o Python instalado. Depois instale as bibliotecas utilizadas por meio do comando:

        pip install -r requirements.txt

5. Para extrair as características através da CNN que serão utilizadas pelos classificadores, execute:

        python extracao_caracteristicas.py


6. Agora, utilizando as características já extraídas, treine e teste o modelo desejado:

        python classificacao.py

7. Além disso, também é possível realizar um teste por meio da combinação de classificadores, para tal:
   
        python3 combinacao_class.py

<br>


## Classificador de K-Vizinhos Mais Próximos (KNN)

Este projeto implementa o algoritmo de K-Vizinhos Mais Próximos (KNN) para classificação supervisionada. O KNN utiliza a distância para fazer previsões sobre o agrupamento de dados, assumindo que amostras semelhantes estão próximas umas das outras. A escolha de ``k`` (número de vizinhos) e a métrica de distância, como a distância euclidiana, impactam significativamente o desempenho do algoritmo.

### Implementação

A implementação é feita em _Python_ e utiliza as bibliotecas _Numpy_, _Pandas_, _SciKit-Learn_ e _Collections_. As principais funções do algoritmo desenvolvido incluem ``fit`` para armazenamento de dados de treinamento e ``predict`` para classificação. Diferentes valores de ``k`` são explorados para entender seu impacto na precisão.

### Resultados

No artigo avalia-se o algoritmo para diferentes valores de ``k`` e porcentagens de dados de treinamento. Os resultados de precisão para vários cenários são visualizados em gráficos, mostrando o impacto de ``k`` e tamanho do conjunto de dados no desempenho da classificação, além de oferecem insights sobre o desempenho em diferentes condições.

Além disso, o relatório destaca a importância de encontrar um equilíbrio entre o número de características e os valores de ``k`` para uma precisão ideal.

Para mais informações sobre os diversos testes realizados, confira o relatório técnico:

[Relatório KNN](./KNN/Report-KNN.pdf)


### Como Executar

1. Clone este repositório:

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Acesse o diretório do projeto:

        cd KNN/

3. Instale as dependências:

        pip install -r requirements.txt

4. Execute o KNN

        python knn.py

<br>


## Classificação Utilizando Centroides do K-Means como Conjunto de Treinamento para o KNN

Este projeto teve como objetivo utilizar os centroides gerados pelo algoritmo _K-Means_ como conjunto de treinamento para o classificador KNN. 

Com tal meta definida, foram extraídas 1000 amostras de treino e teste, cada uma contendo 132 características dos dígitos de 0 a 9. As amostras foram normalizadas e submetidas à redução de dimensionalidade usando a Análise dos Componentes Principais (_PCA_), resultando em duas dimensões.

Em seguida, uma quantidade variável de centroides foi gerada através do K-Means para ser o conjunto de treinamento do KNN, configurado com ``k = 5``.

No entanto, os resultados obtidos foram de baixa precisão, possivelmente devido à redução de dimensionalidade e a possíveis erros na conversão entre os conjuntos de treinamento e teste.

Para entender melhor os possíveis motivos dos resultados pouco precisos e outras minúcias deste trabalho, confira o relatório técnico:

[Relatório K-means](./K-means/K-Means_Report.pdf)

### Como Executar

1. Clone este repositório:

        git clone https://github.com/GustavoMartinx/Artificial-Intelligence.git

2. Acesse o diretório do projeto:

        cd K-means/

3. Instale as dependências:

        pip install -r requirements.txt

4. Execute o K-means

        python kmeans.py

<br>



## :mortar_board: Autores

<table style="flex-wrap: wrap; display: flex; align-items: center;  flex-direction: column;" ><tr>


<td align="center"><a href="https://github.com/eniira">
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/102331777?v=4" width="100px;" alt=""/>
<br />
 <b>Catarine<br>Cruz
</b>
 </a> <a href="https://github.com/eniira" title="Repositorio Catarine Cruz"></a>
</td>

<td align="center"><a href="https://github.com/Fgarm">
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/69016293?v=4" width="100px;" alt=""/>
<br />
 <b>Guilherme<br>Maturana</b></a>
 <a href="https://github.com/Fgarm" title="Repositorio Guilherme Maturana"></a>
</td>

<td align="center"><a href="https://github.com/GustavoMartinx">
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/90780907?v=4" width="100px;" alt=""/>
<br />
 <b>Gustavo<br>Martins</b>
 </a> <a href="https://github.com/GustavoMartinx" title="Repositorio Gustavo Martins"></a>
</td>

<td align="center"><a href="https://github.com/RenanGAS">
 <img style="border-radius: 50%;" src="https://avatars.githubusercontent.com/u/68087317?v=4" width="100px;" alt=""/>
<br />
 <b>Renan<br>Sakashita
</b>
 </a> <a href="https://github.com/RenanGAS" title="Repositorio Renan Sakashita"></a>

</td>

</tr></table>