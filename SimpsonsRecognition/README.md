<h1 align="center"> Reconhecimanto das Personagens da Animação "Os Simpsons" </h1>

Escolhemos como tema do nosso projeto final a implementação de um sistema inteligente que efetua o reconhecimento das personagens principais da animação "Os _Simpsons_" em imagens por meio de técnicas de Aprendizagem de Máquina.

Para tal, utilizou-se uma Rede Neural Convolocional (CNN) a fim de realizar a extração de características, além de diferentes classificadores para avaliação do conjunto de dados, foram eles: K-vizinhos mais próximos, Máquina de Vetores de Suporte (SVM), Perceptron Multicamadas (MLP) e Combinação de Classificadores.

A partir desses métodos, atingimos bom desempenho com taxa de acerto (F1 score) de 74% para o modelo de redes neurais artificiais (o MLP) e 79% para o SVM com a utilização da técnica de _data augmentation_.

Em suma, o reconhecimento das personagens não é uma tarefa trivial. Sob essa ótica, mesmo que o uso de mais dados para treinamento possa aumentar consideravelmente a precisão, traz consigo, infelizmente, um aumento significativo do custo computacional. Em caso de interesse sobre tal trabalho, recomenda-se a leitura de seu respectivo artigo na íntegra:

[Sistema Inteligente para Reconhecimento das Personagens da Animação "Os Simpsons"](./Artigo_Simpsons.pdf)


## Como Executar

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