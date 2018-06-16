import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
import cv2
import os
import itertools
BASE_PATH = "/data/nvidia-docker/data";
data_dir = os.path.join(BASE_PATH,"UCF-101")
classMapFile = os.path.join(BASE_PATH,"ucfTrainTestlist/classInd.txt")

def getVgg(sess,imPh):
    img = imPh
    tf.keras.backend.set_session(sess)
    vgg = tf.keras.applications.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=img,
        input_shape=None,
        pooling=None,
        classes=1000
        )
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
            videoFile = os.path.join(label_dir, video)
            filenames.append(videoFile)
            labels.append(int(label_dic[label_name]) - 1)
    return filenames, labels

def extendModel(cutLayer,x):
    input = cutLayer
    rn_number = int(x.shape[1])
    feature_number = int(input.shape[1])
    w_in = tf.get_variable("w_in",[feature_number,rn_number])
    w_x = tf.get_variable("w_x",[rn_number,rn_number])
    out_in = tf.matmul(input,w_in)
    out_in
    x
    out_x = tf.matmul(x,w_x)
    out = tf.nn.relu(out_in+out_x)
    return out

from tensorflow.python.ops import init_ops




#
# file
# file =videos[0][0]
# im = cv2.imread(file)
# im = im[0:224,0:224,:]
# plt.imshow(im)
# print(im.shape)
# print(x.shape)
# im = np.expand_dims(im,0)
# print(file )



# imTensor = _parse_function(filename[0],224)
# imTensor
# sess.run(imTensor)
# sess.run(vgg)
#
#
# tf.reset_default_graph()
# sess.close()
# image = sess.run(tf.expand_dims(imTensor,0))
# image = sess.run(imTensor)
#
#
# res = sess.run(vgg.output,{x: image})
# res = vgg.predict(image)
#
#
# tf.global_variables()

# tf.summary.FileWriter('/data/tgraph', sess.graph)

def resizeFrame(frame,w=224,h=224):
    cutw = int((frame.shape[1]-w)/2)
    cuth = int((frame.shape[0]-w)/2)
    return frame[cuth:cuth+h,cutw:cutw+w,:]



# input = vggCutLayer
# x = xPh
# name = "reservoir"

def reservoir(input,x,name="reservoir"):
    initializer = init_ops.random_normal_initializer()
    dtype = tf.float32
    rn_number = int(x.shape[1])
    input_size = int(input.output_shape[1])
    print("Reservoir input size: {}".format(input_size))
    with tf.variable_scope(name):  # "ESNCell"
        w_in = tf.get_variable("InputMatrix", [input_size, rn_number], dtype=dtype,
                            trainable=False, initializer=initializer)
        w_r = tf.get_variable("ReservoirMatrix", [rn_number, rn_number], dtype=dtype,
                           trainable=False, initializer=initializer)
        # in_mat = tf.concat([input, state], axis=1)
        # weights_mat = tf.concat([win, wr], axis=0)
        out_in = tf.matmul(input.output,w_in)
        out_x = tf.matmul(x,w_r)
        output = tf.nn.relu(out_in+out_x)
        # output = (1 - self._leaky) * state + self._leaky * self._activation(math_ops.matmul(in_mat, weights_mat) + b)
        return output


def constuctGraph(sess,rn_number = 1600):
    xPh= tf.placeholder(tf.float32,[None,rn_number],name="prediction")
    imPh = tf.placeholder(tf.float32, shape=(None,224, 224,3),name="cnnInput")
    vgg = getVgg(sess,imPh)
    vggCutLayer = vgg.layers[-3] # first fully connected layer
    feature_number = vggCutLayer.output.shape[1]
    print("Featur number of cutLayer {}".format(feature_number))
    model = reservoir(vggCutLayer,xPh)
    return model,vggCutLayer,imPh,xPh
# sess = tf.Session()

videos,labels = dirToVideoLabel(data_dir,labelDicFromFile(classMapFile))

with tf.Session() as sess:
    rn_number = 1600
    model,vggCutLayer,imPh,xPh = constuctGraph(sess,rn_number)
    sess.run(tf.global_variables_initializer())
    videoVectors = []
    for videopath,label in zip(videos,labels):
        print("Processing video {} with label {}".format(videopath,label))
        xPrediction = np.zeros((1,rn_number))
        xMean = np.zeros_like(xPrediction)
        featureMean = np.zeros((1,vggCutLayer.output.shape[1]))
        video_capture = cv2.VideoCapture(videopath)
        success, frame = video_capture.read()
        # frameId = int(video_capture.get(1))
        frameNum = 0
        while success:
            frameNum = frameNum + 1
            frame = resizeFrame(frame)
            frame = np.array([frame])
            # filename = "{}_{}.jpg".format(rootname, str(frameId))
            print("converting frame {}".format(frameNum))
            featureVector, xPrediction = sess.run([vggCutLayer.output,model],feed_dict={xPh:xPrediction,imPh:frame})
            xMean = xMean + xPrediction
            featureMean = featureMean + featureVector
            # cv2.imwrite(os.path.join(dest, filename), img=image)
            success, frame = video_capture.read()
            # frameId = int(video_capture.get(1))
        videoVectors.append(np.concatenate(featureMean/frameNum,xMean/frameNum,axis=1))
    npVideoVectors = np.array(videoVectors)
    np.save("/data/UCFvectors",npVideoVectors)
