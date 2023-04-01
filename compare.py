import numpy as np
import cv2
import matplotlib.pyplot as plt

for i in range(6):
    f1 = f'left-{i}-text.png'
    f2 = f'p{i}.png'
    
    i1 = np.bitwise_and(cv2.imread(f1), 0x80).astype(int)
    i2 = np.bitwise_and(cv2.imread(f2), 0x80).astype(int)
    
    plt.subplot(1, 6, i + 1)
    d = (i1 - i2)
    print(np.unique(d))
    mask = np.where(d == 0)
    d =(d + 10) / 20
    # d[mask] = 0.5
    # print(np.unique(d))
    plt.imshow(d)
    plt.axis('off')
plt.show()
    
    
    