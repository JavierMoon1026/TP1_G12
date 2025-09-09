import numpy as np
import cv2

def Escala_de_grises(img_path):
    return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

def Separar_en_bloques(matriz_img, tam_bloque):
    bloques=[]
    alto,ancho=matriz_img.shape
    for i in range(0,alto,tam_bloque):
        for j in range(0,ancho, tam_bloque):
            bloque = matriz_img[i:i+tam_bloque, j:j+tam_bloque]
            if bloque.shape == (tam_bloque, tam_bloque):
                bloque=bloque.reshape(-1,1,order='F')
                bloques.append(bloque)
    return np.array(bloques)

def pca_transform(X_centrado,porcentaje_de_espacio):
    _,dim_vector,_ =X_centrado.shape
    k=(porcentaje_de_espacio/100)-1
    k*=-dim_vector
    k=round(k)
    X2D = X_centrado.squeeze(-1)  
    U, S, VT = np.linalg.svd(X2D, full_matrices=False)
    U_k= U[:,:k]
    S_k=np.diag(S[:k])
    VT_k = VT[:k, :]        
    return U_k,S_k,VT_k
    


def comprimir(img_path, tam_bloque):
    Imagen_escalada=Escala_de_grises(img_path)
    X=Separar_en_bloques(Imagen_escalada,tam_bloque)
    Media_X = np.mean(X, axis=1, keepdims=True)  
    X_centrado = X - Media_X
    U_k,S_k,VT_k=pca_transform(X_centrado,80)
    Y_k = U_k @ S_k  
    
    return Y_k, Media_X, VT_k

def descomprimir(matriz_Y_k,Media_X, VT_k):
    Xreconstruida=(np.array(matriz_Y_k)@np.array(VT_k))[:,:, None]+np.array(Media_X)
    for vector in Xreconstruida:
        vector = vector.reshape(8, 8, order='F')
    return Xreconstruida
    



path=r"img_01.jpg"

# Comprimir la imagen en las 3 matrices y, mu, vt
y,mu,vt=comprimir(path,8)

# Descomprimir la imagen y reconstruir la matriz de bloques
X_reconstruida = descomprimir(y, mu, vt)

# Reconstruir la imagen completa a partir de los bloques
def reconstruir_imagen(bloques, tam_bloque, alto, ancho):
    img_rec = np.zeros((alto, ancho), dtype=np.uint8)
    idx = 0
    for i in range(0, alto, tam_bloque):
        for j in range(0, ancho, tam_bloque):
            if idx < len(bloques):
                bloque = bloques[idx].reshape(tam_bloque, tam_bloque, order='F')
                img_rec[i:i+tam_bloque, j:j+tam_bloque] = np.clip(bloque, 0, 255)
                idx += 1
    return img_rec

# Obtener dimensiones originales
Imagen_escalada = Escala_de_grises(path)
alto, ancho = Imagen_escalada.shape

# Reconstruir y mostrar la imagen
img_final = reconstruir_imagen(X_reconstruida, 8, alto, ancho)
cv2.imshow('Imagen Reconstruida', img_final.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


