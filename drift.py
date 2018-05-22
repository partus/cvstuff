import tensorflow as tf
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
tf.unpack

def _parse_function(filename, size):
    """Obtain the image from the filename (for both training and validation).

    The following operations are applied:
        - Decode the image from jpeg format
        - Convert to float and to range [0, 1]
    """
    print("parse")
    image_string = tf.read_file(filename)
    # Don't use tf.image.decode_image, or the output shape will be undefined

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)

    # Randomly crop an image with size*size
    image = tf.random_crop(value=image, size=[size, size, 3])

    return image



x = tf.placeholder(tf.float32, shape=(None,224, 224,3))
vgg = tf.keras.applications.VGG16(
    include_top=True,
    weights='imagenet',
    input_tensor=x,
    input_shape=None,
    pooling=None,
    classes=1000
)
tf.keras.backend.set_session(sess)
vgg
tf.keras.backend.clear_session()

del vgg

import os
BASE_PATH = "/data/nvidia-docker/data";
data_dir = os.path.join(BASE_PATH,"UCF-101_train01")
classMapFile = os.path.join(BASE_PATH,"ucfTrainTestlist/classInd.txt")

c = tf.constant([5.0,6.0])
c

sess.run(c)

def labelDicFromFile(name):
    label_dic = {}
    with open(name) as f:
        for line in f:
            (val, key) = line.split()
            label_dic[key] = int(val)
    return label_dic

def dirToPathLabel(data_dir,label_dic):
    labels = []
    filenames = []
    for label_name in os.listdir(data_dir):
        lab_dir = os.path.join(data_dir, label_name)
        for video_dir in os.listdir(lab_dir):
            vid_dir = os.path.join(lab_dir, video_dir)
            for f in os.listdir(vid_dir):
                filenames.append(os.path.join(vid_dir, f))
                labels.append(int(label_dic[label_name]) - 1)
    return filenames, labels

filename[0]
filename,labels = dirToPathLabel(data_dir,labelDicFromFile(classMapFile))



imTensor = _parse_function(filename[0],224)
imTensor
sess.run(imTensor)
sess.run(vgg)
sess = tf.Session()
tf.reset_default_graph()
sess.close()
image = sess.run(tf.expand_dims(imTensor,0))
image = sess.run(imTensor)
image.shape
np.ndarray(image)
sess.run(tf.global_variables_initializer())
res = sess.run(vgg.output,{x: image})
res = vgg.predict(image)
res.shape
plt.hist(res)

tf.global_variables()
plt.hist([1,2])
res.shape

res
sess.run(x,{x:image})
tf.summary.FileWriter('/data/tgraph', sess.graph)
