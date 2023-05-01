from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import pandas as pd
import numpy as np
import os

class Extraction():
    
    blackPixels = 0
    whitePixels = 0

    # Treino
    # Criando arquivo txt
    with open("Treino_3x3.csv", "a") as res_Treino_3x3:

        # Current Working Directory
        project = os.getcwd()

        # percorre pelas pastas de numeros (classes) dentro de treino
        dirTreino = project + "/treino/"

        # (k^2) *2  | 1x1 = 1 | 2x2 = 7 | 3x3 = 19 | 5x5 = 49
        cols = ""
        cols = cols + "Col1"
        for i in range(18):
            cols = cols + f",Col{str(i+2)}"
        res_Treino_3x3.write((cols + "\n"))

        for j in range(10):
            numberDir = dirTreino + str(j)

            # iterate over files in that directory
            for filename in os.scandir(numberDir):
                if filename.is_file():
                
                    # logica a ser aplicada a cada arquivo aqui

                    # abrindo imagem
                    img = Image.open(filename.path) # "./treino/0/37138-2-4.bmp"

                    # obtendo as dimensões de uma imagem
                    imgWidth = img.size[0]      # largura
                    imgHeight = img.size[1]     # altura
                    quadrantes = []

                    # print("Inteira 1x1 imgWidth: ", imgWidth)
                    # print("Inteira 1x1 imgHeight: ", imgHeight)


                    # obtendo as dimensões dos quadrantes (3x3)
                    for i in range(3):
                        for z in range(3):
                            x0 = imgWidth // 3 * i
                            y0 = imgHeight // 3 * z
                            if i == 2:
                                x1 = imgWidth
                            else:
                                x1 = x0 + imgWidth // 3
                            if z == 2:
                                y1 = imgHeight
                            else:
                                y1 = y0 + imgHeight // 3
                            
                            quadrante = img.crop((x0, y0, x1, y1))
                            # print(f"Largura Altura do Quadrante {i}x{j}: {quadrante.size}")
                            quadrantes.append(quadrante)

                    # percorrendo a imagem dividida por quadrantes (3x3)
                    for quadrante in quadrantes:
                        q_largura, q_altura = quadrante.size
                        
                        # zerando a quantidade de pixels a cada quadrante
                        blackPixels = 0
                        whitePixels = 0
                        
                        for x in range(q_largura):
                            for y in range(q_altura):
                                pixelValue = quadrante.getpixel((x, y))
                                # se o pixel for preto
                                if pixelValue == 0:
                                    blackPixels += 1
                                elif pixelValue == 255:
                                    whitePixels += 1

                        # printar a quantidade pretos e brancos de cada quadrante
                        res_Treino_3x3.write(f"{blackPixels},{whitePixels},")
                    
                    res_Treino_3x3.write(f"{str(j)}\n")


    # Normalizando dados de Treino
    df_treino = pd.read_csv("Treino_3x3.csv")

    # print(df_treino)
    # print(df_treino.describe())

    features = df_treino.iloc[:, :-1].values
    classes = df_treino.iloc[:, -1].values

    # print(features)
    # print(classes)

    # Criando o objeto normalizer
    normalizer = MinMaxScaler()

    # Ajustando e aplicando a transformação
    minmax = normalizer.fit_transform(features)

    # Concatenando o min-max com a classe
    data_minmax = np.hstack((minmax, classes.reshape(-1, 1)))

    # print("-----")
    # print(data_minmax)

    # Salvando a base de dados normalizada
    np.savetxt('Treino_3x3_normalizado.txt', data_minmax, delimiter=',', header=cols, fmt='%f', comments='')




    # Teste
    with open("Teste_3x3.csv", "a") as res_Teste_3x3:

        # percorre pelas pastas de numeros dentro de teste
        dirTeste = project + "/teste/"

        # (k^2) *2  | 1x1 = 1 | 2x2 = 7 | 3x3 = 17 | 5x5 = 49
        cols = ""
        cols = cols + "Col1"
        for i in range(18):
            cols = cols + f",Col{str(i+2)}"
        res_Teste_3x3.write((cols + "\n"))

        for i in range(10):
            numberDir = dirTeste + str(i)

            # iterate over files in that directory
            for filename in os.scandir(numberDir):
                if filename.is_file():
                    # logica a ser aplicada a cada arquivo aqui
                    
                    # abrindo imagem
                    img = Image.open(filename.path) # "./treino/0/37138-2-4.bmp"

                    # obtendo as dimensões de uma imagem
                    imgWidth = img.size[0]      # largura
                    imgHeight = img.size[1]     # altura
                    quadrantes = []

                    # print("Inteira 1x1 imgWidth: ", imgWidth)
                    # print("Inteira 1x1 imgHeight: ", imgHeight)

                    # obtendo as dimensões dos quadrantes (3x3)
                    for k in range(3):
                        for j in range(3):
                            x0 = imgWidth // 3 * k
                            y0 = imgHeight // 3 * j
                            if k == 2:
                                x1 = imgWidth
                            else:
                                x1 = x0 + imgWidth // 3
                            if j == 2:
                                y1 = imgHeight
                            else:
                                y1 = y0 + imgHeight // 3
                            
                            quadrante = img.crop((x0, y0, x1, y1))
                            # print(f"Largura Altura do Quadrante {k}x{j}: {quadrante.size}")
                            quadrantes.append(quadrante)

                    # percorrendo a imagem dividida por quadrantes (3x3)
                    for quadrante in quadrantes:
                        q_largura, q_altura = quadrante.size
                        
                        # zerando a quantidade de pixels a cada quadrante
                        blackPixels = 0
                        whitePixels = 0
                        
                        for x in range(q_largura):
                            for y in range(q_altura):
                                pixelValue = quadrante.getpixel((x, y))
                                # se o pixel for preto
                                if pixelValue == 0:
                                    blackPixels += 1
                                elif pixelValue == 255:
                                    whitePixels += 1

                        
                        # printar a quantidade pretos e brancos de cada quadrante
                        res_Teste_3x3.write(f"{blackPixels},{whitePixels},")
                    
                    # Escrevendo a classe que aquela imagem pertence no fim de cada linha
                    res_Teste_3x3.write(f"{str(i)}\n")
                    
    # Normalizando dados de Teste
    df_teste = pd.read_csv("Teste_3x3.csv")

    # print(df_teste)
    # print(df_teste.describe())

    features = df_teste.iloc[:, :-1].values
    classes = df_teste.iloc[:, -1].values

    # print(features)
    # print(classes)

    # Criando o objeto normalizer
    normalizer = MinMaxScaler()

    # Ajustando e aplicando a transformação
    minmax = normalizer.fit_transform(features)

    # Concatenando o min-max com a classe
    data_minmax = np.hstack((minmax, classes.reshape(-1, 1)))

    # print("-----")
    # print(data_minmax)

    # Salvando a base de dados normalizada
    np.savetxt('Teste_3x3_normalizado.txt', data_minmax, delimiter=',', header=cols, fmt='%f', comments='')
