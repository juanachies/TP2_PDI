import cv2
import numpy as np
import matplotlib.pyplot as plt

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


for i in range(1,3):
    img = cv2.imread(f'img{i:02d}.png')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)

    binary = cv2.adaptiveThreshold(
        norm, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        21, 10
    )

    # v = np.median(binary)
    # low = int(1 * v)
    # high = int(1.5 * v)
    # edges = cv2.Canny(binary, low, high, apertureSize=3, L2gradient=True)

    imshow(binary)

    L = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (L, L) )
    morph = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    imshow(morph)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    img_contours = img.copy()
    cv2.drawContours(img_contours, contours[:10], -1, (0, 255, 0), 2)

    imshow(img_contours)