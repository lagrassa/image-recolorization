import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import pdb
import sys

def main(file0):
    img = cv2.imread(file0)
    color = ('b','g','r')
    plt.figure()
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}
    matplotlib.rc('font', **font)
    plt.axis([0, 255, 0, 8000])
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        AUC = sum(np.asarray(histr[-8:], dtype=float))
        print("AUC "+ str(AUC))

        plt.plot(histr,color = col, linewidth = 10)
        plt.xlim([0,256])
    plt.show()

if __name__ == "__main__":
    file0 = sys.argv[1]
    main(file0)

