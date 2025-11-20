import cv2
import numpy as np
import matplotlib.pyplot as plt

# Segmentar con canny o umbralizado.
# Hay que implementar un for que ajuste los parámetros según la imagen (100%)

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)


for i in range(1,4):
    
    # Cargamos la imagen
    img = cv2.imread(f'img0{i}.png')

    # Canny
    v = np.median(img)
    low = int(0.66 * v)
    high = int(1.33 * v)
    edges = cv2.Canny(img, low, high, apertureSize=3, L2gradient=True)

    #img_canny_CV2 = cv2.Canny(img, 150, 255, apertureSize=3, L2gradient=True)

    output = img.copy()
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.drawContours(output, [cnt], -1, (255,0,0), 2)
        #cv2.putText(output, f'{label}' , (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

    
    # Cierre morfológico 
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    #closed = cv2.morphologyEx(img_canny_CV2, cv2.MORPH_CLOSE, kernel)

    imshow(edges)