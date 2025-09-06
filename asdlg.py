import numpy as np
import cv2


def dividir_en_bloques(img_array, tam_bloque=8):
    columnas, filas = img_array.shape
    bloques = []
    for i in range(0, columnas, tam_bloque):
        for j in range(0, filas, tam_bloque):
            bloque = img_array[i:i+tam_bloque, j:j+tam_bloque]
            if bloque.shape == (tam_bloque, tam_bloque):
                bloques.append(bloque)
    return bloques



def pca_transform(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Vector_Aleatorio_X = dividir_en_bloques(img_gray)
    Esperanza_de_Vector_Aleatorio_X=[]
    for bloque in Vector_Aleatorio_X:
        u=0
        for columna in bloque:
            for fila in columna:
                u+=float(fila)
        Esperanza_de_Vector_Aleatorio_X.append(u/(len(columna)*len(bloque)))

    matriz_Centrada_A= np.array(Vector_Aleatorio_X) - np.array(Esperanza_de_Vector_Aleatorio_X)[:,None,None]
    U,S,VT= np.linalg.svd(matriz_Centrada_A)
    print()




pca_transform('img_01.jpg')