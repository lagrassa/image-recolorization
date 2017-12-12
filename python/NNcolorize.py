import tensorflow as tf
import sys
import pdb
import richzhangAPI
from utils import *
import autocolorize
from net import Net
from skimage.io import imsave
from skimage.transform import resize
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

DEBUG = False

def normalize(img):
    img = img[:,:,:3] #remove alpha channel (confirmed that it looks OK)
    img = img/255.0 #and convert to that 0,1 space
    return img

def get_auc(img):
    auc = 0
    color = ('b','g','r')  
    for i, col in enumerate(color):
        histr = np.histogram(img[:,:,i],bins=30)[1]
        cutoff = 3
        histr = histr[cutoff:len(histr)-cutoff]
        #real looking images have good intermediate values, so throw away the end values
        auc += sum(histr)
        print("The AUC now is")
    return auc



#kinda hard to quantify. Metrics are
#1. saturation
#2. How real the coloring looks
def choose_best_rgb(outputs, filename):
    scores = []
    i = 0
    funcs = ["tfnet", "tfnet_mine", "autocolor", "richzhang"]
    for output in outputs:
        score = 0
        hsv_img = matplotlib.colors.rgb_to_hsv(output)
        sat = hsv_img[:,:,1] 
        auc = get_auc(output)
        score += int(sum(sum(sat)))
        score += auc
        #evaluate saturation
        #evaluate how real the coloring is
        scores.append(score)
        if DEBUG:
            savename = '../results/'+funcs[i]+filename
            matplotlib.pyplot.imsave(savename, output)
        i +=1
    return plot_and_choose_im(outputs, scores)

def plot_and_choose_im(outputs, scores):
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
    matplotlib.pyplot.show(block=False)
    best = int(input("Which number is the best colorization?"))
    return outputs[best-1]

    




def get_all_nn_outputs(img,img_name):
    print("getting all nn outputs")
    outputs = []
    funcs = [tf_net, tf_net_mine, autocolor, richzhang]
    for func in funcs:
        if func != richzhang:
            colored_image = func(img)
        else:
            colored_image = func(img_name)
        outputs.append(colored_image)
    return outputs

def main(filename, toMATLAB):
    img = cv2.imread(filename)
    outputs = get_all_nn_outputs(img, filename)
    best_rgb = choose_best_rgb(outputs, filename)
    if not toMATLAB:
        modified_filepath = mod_filepath(filename)
        filepath = modified_filepath
    else:
        filepath = "../matlab/nnoutput.jpg"
    pdb.set_trace()
    imsave(filepath, best_rgb)
    #return modified_filepath


    
def richzhang(img):
    return richzhangAPI.colorize(img)
    
def autocolor(img):
    img = img/255
    #if len(img.shape) == 3:
    #  img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    classifier = autocolorize.load_default_classifier()
    rgb = autocolorize.colorize(img, classifier=classifier)
    return rgb


def tf_net_mine(img):
    #img = img[:,:,:3] #remove alpha channel (confirmed that it looks OK)
    #img = img/255.0 #and convert to that 0,1 space
    if len(img.shape) == 3:
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = img[None, :, :, None]
    data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    #data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    tf.reset_default_graph() 
    autocolor = Net(train=False)

    conv8_313 = autocolor.inference(data_l)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, 'models/model_mine.ckpt')
      conv8_313 = sess.run(conv8_313)

    img_rgb = decode(data_l, conv8_313,2.63)

    return img_rgb

def tf_net(img):
    #img = img[:,:,:3] #remove alpha channel (confirmed that it looks OK)
    #img = img/255.0 #and convert to that 0,1 space
    if len(img.shape) == 3:
        img = np.array(img, dtype=np.uint8)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    img = img[None, :, :, None]
    data_l = (img.astype(dtype=np.float32)) / 255.0 * 100 - 50

    #data_l = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    autocolor = Net(train=False)

    conv8_313 = autocolor.inference(data_l)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      saver.restore(sess, 'models/model.ckpt')
      conv8_313 = sess.run(conv8_313)

    img_rgb = decode(data_l, conv8_313,2.63)

    return img_rgb


def mod_filepath(filename):
    last_dot = filename.rfind('.')
    mod_name =  filename[:last_dot]+"_nncolor"+filename[last_dot:]

    return mod_name

if __name__ == '__main__':
    gray_img = sys.argv[1]
    main(gray_img, True)



