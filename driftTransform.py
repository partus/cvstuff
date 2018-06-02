import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import os
import itertools
BASE_PATH = "/data/nvidia-docker/data";
data_dir = os.path.join(BASE_PATH,"UCF-101")
classMapFile = os.path.join(BASE_PATH,"ucfTrainTestlist/classInd.txt")


def getVgg(sess,imPh):
    img = imPh
    vgg = tf.keras.applications.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=img,
        input_shape=None,
        pooling=None,
        classes=1000
        )
    tf.keras.backend.set_session(sess)
    return vgg

# tf.keras.backend.clear_session()

def labelDicFromFile(name):
    label_dic = {}
    with open(name) as f:
        for line in f:
            (val, key) = line.split()
            label_dic[key] = int(val)
    return label_dic

def dirToVideoLabel(data_dir, label_dic):
    labels = []
    filenames = []
    for label_name in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label_name)
        for video in os.listdir(label_dir):
            videoFile = os.path.join(lab_dir, video)
            filenames.append(videoFile)
            labels.append(int(label_dic[label_name]) - 1)
    return filenames, labels

def extendModel(nn,x):
    input = nn.output
    rn_number = int(x.shape[1])
    feature_number = int(vgg.output.shape[1])
    w_in = tf.get_variable("w_in",[feature_number,rn_number])
    w_x = tf.get_variable("w_x",[rn_number,rn_number])
    out_in = tf.matmul(input,w_in)
    out_in
    x
    out_x = tf.matmul(x,w_x)
    out = tf.nn.relu(out_in+out_x)
    return out



file
file =videos[0][0]
im = cv2.imread(file)
im = im[0:224,0:224,:]
plt.imshow(im)
print(im.shape)
print(x.shape)
im = np.expand_dims(im,0)
print(file )



imTensor = _parse_function(filename[0],224)
imTensor
sess.run(imTensor)
sess.run(vgg)


tf.reset_default_graph()
sess.close()
image = sess.run(tf.expand_dims(imTensor,0))
image = sess.run(imTensor)


res = sess.run(vgg.output,{x: image})
res = vgg.predict(image)


tf.global_variables()

# tf.summary.FileWriter('/data/tgraph', sess.graph)


videos,labels = dirToVideoLabel(data_dir,labelDicFromFile(classMapFile))
with sess as tf.Session():
    rn_number = 1600
    xPh= tf.placeholder(tf.float32,[None,rn_number],name="prediction")
    imPh = tf.placeholder(tf.float32, shape=(None,224, 224,3),name="cnnInput")
    vgg = getVgg(sess,imPh)
    model =  extendModel(vgg,xPh)
    sess.run(tf.global_variables_initializer())
    for video,label in itertools.izip(videos,labels):
        xPrediction = np.zeros((1,rn_number))
        cv2.video_capture
                    frame = cv2.imread(framepath)
                    xPrediction = sess.run(model,feed_dict={xPh:xPrediction,imPh:im})
                xPath = ""
                np.save(xPath,xPrediction)
