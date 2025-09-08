import numpy as np
import matplotlib.pyplot as plt
import cv2

def img_to_vec(img):
    ImgFloat = img.astype(float)
    ImgVec = ImgFloat.T.ravel()
    x1 = ImgVec[::2]
    x2 = ImgVec[1::2]
    return np.vstack([x1, x2]).T

def graficar(x, img, titulo="Imagen"):
    x1 = x[:,0]
    x2 = x[:,1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(titulo)

    axes[1].scatter(x1, x2, s=1, alpha=0.5)    
    axes[1].set_xlabel("x1 (pixel de arriba)")   
    axes[1].set_ylabel("x2 (pixel de abajo)") 
    axes[1].set_title("Dispersion de pixeles vecinos")
    
    plt.tight_layout()
    plt.show()

    corr = np.corrcoef(x1, x2) [0, 1]
    print(corr)

def descorrelacion(x):
    mu = x.mean(axis=0)
    x_centrado = x - mu

    cov = np.cov(x_centrado, rowvar=False)

    eigval, eigvec = np.linalg.eigh(cov)

    idx = np.argsort(eigval)[::1]
    eigvec = eigvec[:, idx]

    return x_centrado @ eigvec

def graficar_descorrelacion(x, titulo="Imagen - Vectores descorrelacionados"):
    plt.figure(figsize=(5,5))
    plt.scatter(x[:,0], x[:,1], s=1, alpha=0.5)
    plt.xlabel("")
    plt.ylabel("")
    plt.title(titulo)
    plt.show()


img1 = cv2.imread("img_01.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("img_02.jpg", cv2.IMREAD_GRAYSCALE)

v1 = img_to_vec(img1)
v2 = img_to_vec(img2)

graficar(v1, img1, "Imagen 1")
graficar(v2, img2, "Imagen 2")

d1 = descorrelacion(img1)
d2 = descorrelacion(img2)

graficar_descorrelacion(d1, "Imagen1 - Vectores descorrelacionados")
graficar_descorrelacion(d2, "Imagen2 - Vectores descorrelacionados")


