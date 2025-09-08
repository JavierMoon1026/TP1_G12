import numpy as np
import cv2


def dividir_en_bloques(img_array, tam_bloque):
    columnas, filas = img_array.shape
    bloques = []
    for i in range(0, columnas, tam_bloque):
        for j in range(0, filas, tam_bloque):
            bloque = img_array[i:i+tam_bloque, j:j+tam_bloque]
            if bloque.shape == (tam_bloque, tam_bloque):
                bloques.append(bloque)
    return bloques

def truncamiento_SVD(realizacion,porcentaje_espacio_ahorrado):
    cantidad = len(realizacion)
    k= (porcentaje_espacio_ahorrado/100)-1
    k*=-(cantidad)
    k=round(k)
    U,S,VT= np.linalg.svd(realizacion)
    U_k = U[:, :k]      
    S_k = S[:k]         
    Vt_k = VT[:k, :]
    return U_k , S_k, Vt_k

def proyeccion_SVD_truncado(U_k,S_k):
    return U_k @ S_k

def pca_transform(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    Vector_Aleatorio_X = dividir_en_bloques(img_gray,8)
    X=[]
    for bloques in Vector_Aleatorio_X:
        X.append(bloques.reshape(-1, order='F'))
    Esperanza_X=[]
    for realizacion in X:
        u=0
        for elemento in realizacion:
            u+=int(elemento)
        Esperanza_X.append(u/(len(realizacion)))
    Matriz_Centrada_A= np.array(X) - np.array(Esperanza_X)[:,None]
    # for elemento in Vector_Aleatorio_X:
    #     u=0
    #     for columna in bloque:
    #         for fila in columna:
    #             u+=float(fila)
    #     Esperanza_X.append(u/(len(columna)*len(bloque)))
    # matriz_Centrada_A= np.array(Vector_Aleatorio_X) - np.array(Esperanza_X)[:,None,None]
    Y=[]
    V_k=[]
    for realizacion in Matriz_Centrada_A:
        U_k , S_k, Vt_k = truncamiento_SVD(realizacion,80)
        V_k.append(Vt_k.T)
        Y.append(proyeccion_SVD_truncado(U_k,S_k))
    Y=np.array(Y)
    return Y, V_k, Esperanza_X

def invertir_pca(matriz_Y, vector_V_k, Esperanza_de_x):
    Esperanza_de_X=Esperanza_de_x.copy()
    algo=[]
    for Yi, V_k_i, ux in zip(matriz_Y,vector_V_k,Esperanza_de_x):
        algo.append((Yi@V_k_i)+ux[:,None])
    print(algo[0])



    

        
    # Y=np.array(Y)
    # print(Y.shape)




y,v,e=pca_transform('img_01.jpg')
invertir_pca(y,v,e)