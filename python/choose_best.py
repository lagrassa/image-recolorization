import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, draw, show
import scipy
from scipy import ndimage

imagenames = ['../results/autocolorlucy.jpg','../results/tfnetlucy.jpg','../results/tfnet_minelucy.jpg','../results/richzhanglucy.jpg']
outputs = [ndimage.imread(i) for i in imagenames]
scores = [100,200000,300,6]



f, axarr = plt.subplots(2,2)
plt.axis('off')
axarr[0,0].imshow(outputs[0])
axarr[0,0].set_title("Colorization 1, Score:"+str(scores[0]))

axarr[0,1].imshow(outputs[1])
axarr[0,1].set_title("Colorization 2, Score:"+str(scores[1]))

axarr[1,0].imshow(outputs[2])
axarr[1,0].set_title("Colorization 3, Score:"+str(scores[2]))

axarr[1,1].imshow(outputs[3])
axarr[1,1].set_title("Colorization 4, Score:"+str(scores[3]))
show(block=False)
best = int(input("Which number is the best colorization?"))
print(str(best) +" is the best image")
plt.imshow(outputs[best])

show()


