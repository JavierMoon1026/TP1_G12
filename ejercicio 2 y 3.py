import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def descomprimir(matriz_Y_k, Media_X, VT_k):
    Xreconstruida = (np.array(matriz_Y_k) @ np.array(VT_k))[:, :, None] + np.array(Media_X)
    n = Xreconstruida.shape[0]
    X_recon = np.zeros((n, 8, 8))
    for i in range(n):
        X_recon[i] = Xreconstruida[i].reshape(8, 8, order='F')
    
    return X_recon
    
def recomponer_imagen(bloques, alto, ancho, tam_bloque=8):

    imagen = np.zeros((alto, ancho))
    bloques_por_fila = ancho // tam_bloque
    for idx, bloque in enumerate(bloques):
        fila = (idx // bloques_por_fila) * tam_bloque
        col = (idx % bloques_por_fila) * tam_bloque
        imagen[fila:fila+tam_bloque, col:col+tam_bloque] = bloque 
    return imagen

def Graficar_comparacion(X_centrado, k=13):
    X2D = X_centrado.squeeze(-1)
    n, m = X2D.shape
    U, S, VT = np.linalg.svd(X2D)
    autovalores = (S**2) / (n - 1) 
    colors = ['tab:blue' if i < k else 'tab:gray' for i in range(len(autovalores))]

    # Gráfico de barras para autovalores
    fig, ax1 = plt.subplots(figsize=(10, 5))
    indices = np.arange(1, len(autovalores) + 1)
    ax1.bar(indices, autovalores, color=colors, alpha=0.8)
    ax1.set_xlabel('Índice del componente (orden descendente)')
    ax1.set_ylabel('Autovalores de $C_X$')
    ax1.set_title(f'Autovalores de $C_X$ (nxm={n}{m}) — primeros {k} conservados')

    # Línea vertical indicando límite k
    ax1.axvline(k + 0.5, color='red', linestyle='--', linewidth=1)
    ax1.text(k + 1, max(autovalores) * 0.9, f'k = {k}', color='red')

    plt.tight_layout()
    plt.show()


path=r"img_04.jpg"

y,mu,vt=comprimir(path,8)
X_reconstruida = descomprimir(y, mu, vt)

Imagen_escalada = Escala_de_grises(path)
alto, ancho = Imagen_escalada.shape

Imagen_escalada = Escala_de_grises(path)
X = Separar_en_bloques(Imagen_escalada, 8)  
X_centrado = X-mu

Graficar_comparacion(X_centrado)

img_final = recomponer_imagen(X_reconstruida,alto,ancho,8)
cv2.imshow('Imagen Reconstruida', img_final.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
