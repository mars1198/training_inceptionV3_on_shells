import pandas as pd 
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
import h5py
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DIR = "dataset/train/"
TEST_DIR =  "dataset/test/"


def get_label(Dir):
    for nextdir in os.listdir(Dir):
        if not nextdir.startswith('.'):
            if nextdir in ['Ginia munda munda']:
                label = 0
            elif nextdir in ['Ginia sublitoralis']:
                label = 1
            elif nextdir in ['Macedopyrgula pavlovici']:
                label = 2
            elif nextdir in ['Macedopyrgula wagneri']:
                label = 3
            elif nextdir in ['Ohridopyrgula charensis']:
                label = 4
            elif nextdir in ['Ohridopyrgula macedonica']:
                label = 5
            elif nextdir in ['Ohridopyrgula svnaum']:
                label = 6
    return nextdir, label



def preprocessing_data(Dir):
    X = []
    y = []
    
    for nextdir in os.listdir(Dir):
        nextdir, label = get_label(Dir)
        temp = Dir + nextdir
        
        for image_filename in tqdm(os.listdir(temp)):
            path = os.path.join(temp + '/' , image_filename)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = skimage.transform.resize(img, (150, 150, 3))
                img = np.asarray(img)
                X.append(img)
                y.append(label)
            
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X,y



def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['Ginia munda munda']:
                label = 0
            elif nextDir in ['Ginia sublitoralis']:
                label = 1
            elif nextDir in ['Macedopyrgula pavlovici']:
                label = 2
            elif nextDir in ['Macedopyrgula wagneri']:
                label = 3
            elif nextDir in ['Ohridopyrgula charensis']:
                label = 4
            elif nextDir in ['Ohridopyrgula macedonica']:
                label = 5
            elif nextDir in ['Ohridopyrgula svnaum']:
                label = 6
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (150, 150, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(299, 299, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

def plot_confusion_matrix(y_true, y_pred, matrix_title):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
    plt.title(matrix_title, fontsize=12)
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90) 
    plt.show()

X_train, y_train = get_data(TRAIN_DIR)
X_test , y_test = get_data(TEST_DIR)

from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)

from keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, epsilon=0.0001, patience=1, verbose=1)




from keras.models import Sequential , Model
from keras.layers import Dense , Activation
from keras.layers import Dropout , GlobalAveragePooling2D
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD , RMSprop , Adadelta , Adam
from keras.layers import Conv2D , BatchNormalization
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.common.image_dim_ordering()
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

X_train=X_train.reshape(X_train.shape[0],150,150,3)
X_test=X_test.reshape(X_test.shape[0],150,150,3)

from keras.applications.inception_v3 import InceptionV3
# create the base pre-trained model
base_model = InceptionV3(weights=None, include_top=False, input_shape=(150, 150, 3))

x = base_model.output
x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(7, activation='sigmoid')(x)

base_model.load_weights("inception_v3_weights.h5")

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

batch_size = 64
epochs = 10

filepath_checkpoint="best.hdf5"
checkpoint = ModelCheckpoint(filepath=filepath_checkpoint, save_weights_only=True,
monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')

history = model.fit(X_train, y_train, validation_data = (X_test , y_test) ,callbacks=[lr_reduce,checkpoint] ,
          epochs=epochs)

model.load_weights(filepath_checkpoint)


pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)

#plot_confusion_matrix(y_true, pred, 'Confusion Matrix')

import time
saved_model_path = "./saved_models/{}".format(int(time.time()))
import tensorflow as tf
tf.keras.experimental.export_saved_model(model, saved_model_path, save_format=None)

