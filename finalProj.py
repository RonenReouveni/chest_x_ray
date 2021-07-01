# READ ME #

#This is a python script. 
#It is meant to be run block by block on a jupyter server within VS Code
#It was slightly too much data to use on COLAB


#import packages
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image

import cv2
from sklearn.metrics import confusion_matrix
import itertools
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import random
from matplotlib import pyplot as plt
from random import sample




#source data 
#https://www.kaggle.com/prashant268/chest-xray-covid19-pneumonia


#establish path
data_path = '/Users/ronenreouveni/Desktop/Xray/data'

#Data Augmentation
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   zoom_range = 0.3, #.2
                                   rotation_range=20, #15
                                   vertical_flip = False,
                                   horizontal_flip = False)

test_datagen = ImageDataGenerator(rescale = 1./255)

#create iterator 
inputShape = (224, 224)
training_set = train_datagen.flow_from_directory(data_path + '/train',
                                                 target_size = inputShape,
                                                 batch_size = 16,
                                                 class_mode = 'categorical',
                                                 shuffle=True)

test_set = test_datagen.flow_from_directory(data_path + '/test',
                                            target_size = inputShape,
                                            batch_size = 16,
                                            class_mode = 'categorical',
                                            shuffle = False)

#plot data augmentation
plt.figure(figsize=(20,10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_set.__getitem__(0)[0][i])

#code to show nice confusion matrix 
#https://www.udemy.com/course/advanced-computer-vision/learn/lecture/9339972?start=60#overview
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

######################################################################################
#VGG16
######################################################################################

#base model 
foundationModel = VGG16(input_shape=(224,224,3), weights='imagenet', include_top=False)

#turn off training 
for layer in foundationModel.layers:
  layer.trainable = False

#connect models 
headModel = foundationModel.output
x = layers.MaxPooling2D((2, 2))(headModel)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(3, activation='softmax')(x)


model = Model(inputs=foundationModel.input, outputs = x)


opt=Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#create callbacks for early stopping and restoring best weights
callBack = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model.fit(training_set,validation_data=test_set,epochs=4, callbacks=[callBack]) 



#get accuracy, plot matrix
preds = model.predict(test_set)
preds = preds.argmax(1)
class_names = ['Covid','Normal','Pneumonia']
acc = accuracy_score(test_set.labels, preds)
cm = confusion_matrix(test_set.labels, preds)
plot_confusion_matrix(cm, class_names)

######################################################################################
#Supporting Code 
######################################################################################

class_dict = {0:'COVID19',
              1:'NORMAL',
              2:'PNEUMONIA'}


#next two functions were taken from this article
#https://www.kaggle.com/prashant268/covid-19-diagnosis-using-x-ray-images
#the second function I changed to take a model so I could compare 
def find_true_class(file_path):
    true_class = None
    if 'COVID19' in file_path:
        true_class = 'COVID19'
    elif 'PNEUMONIA' in file_path:
        true_class = 'PNEUMONIA'
    elif 'NORMAL' in file_path:
        true_class = 'NORMAL'
    return true_class

#plot lime
def visualize(myModel, file_path,ax,text_loc):
    test_image = cv2.imread(file_path)
    test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
    test_image = np.expand_dims(test_image,axis=0)
    probs = myModel.predict(test_image)
    pred_class = np.argmax(probs)
    pred_class = class_dict[pred_class]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(test_image[0], myModel.predict, top_labels=5, hide_color=0, num_samples=10)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)

    ax.imshow(mark_boundaries(temp, mask))
    fig.text(text_loc, 0.9, "Predicted Class: " + pred_class , fontsize=13)
    true_class = find_true_class(file_path)
    if true_class is not None:
        fig.text(text_loc, 0.86, "Actual Class: " + true_class , fontsize=13)


#find where we were wrong 
wrongOnes = np.where(test_set.labels != preds)

#find the path
wrongPredPath = []
for location in wrongOnes[0]:
   spot = test_set.filepaths[location]
   wrongPredPath.append(spot)

#sample the paths and plot 
x,y,z = sample(wrongPredPath,3)
fig,ax = plt.subplots(1,3,figsize=(18,6))
visualize(model,x,ax[0],0.15)
visualize(model,y,ax[1],0.4)
visualize(model,z,ax[2],0.7)


rightOnes = np.where(test_set.labels == preds)

rightOnesPath = []
for location in rightOnes[0]:
   spot = test_set.filepaths[location]
   rightOnesPath.append(spot)


x,y,z = sample(rightOnesPath,3)
fig,ax = plt.subplots(1,3,figsize=(18,6))
visualize(model,x,ax[0],0.15)
visualize(model,y,ax[1],0.4)
visualize(model,z,ax[2],0.7)




######################################################################################
#DNN
######################################################################################


#create DNN
i = layers.Input(shape = (224,224,3))
x = layers.Flatten()(i)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(3, activation='softmax')(x) 


model_ann = Model(i,x)
opt=Adam(learning_rate=0.0001)
model_ann.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#create callbacks for early stopping and restoring best weights
callBack = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model_ann.fit(training_set,validation_data=test_set,epochs=8, callbacks=[callBack])   


preds_ann = model_ann.predict(test_set)
preds_ann = preds_ann.argmax(1)
cm = confusion_matrix(test_set.labels, preds_ann)
plot_confusion_matrix(cm, class_names)


wrongOnes_ann = np.where(test_set.labels != preds_ann)

wrongPredPath_ann = []
for location in wrongOnes_ann[0]:
   spot = test_set.filepaths[location]
   wrongPredPath_ann.append(spot)

x,y,z = sample(wrongPredPath_ann,3)
fig,ax = plt.subplots(1,3,figsize=(18,6))
visualize(model_ann, x,ax[0],0.15)
visualize(model_ann, y,ax[1],0.4)
visualize(model_ann, z,ax[2],0.7)

rightOnes_ann = np.where(test_set.labels == preds_ann)

rightOnesPath_ann = []
for location in rightOnes_ann[0]:
   spot = test_set.filepaths[location]
   rightOnesPath_ann.append(spot)


x,y,z = sample(rightOnesPath_ann,3)
fig,ax = plt.subplots(1,3,figsize=(18,6))
visualize(model_ann, x,ax[0],0.15)
visualize(model_ann, y,ax[1],0.4)
visualize(model_ann, z,ax[2],0.7)


######################################################################################
#Ronen CNN
######################################################################################

#tried using my CNN architecture from lab 9, does not yield best results and takes a very long time to train
i = layers.Input(shape = (224,224,3))



x = layers.Conv2D(64, (3,3), strides = (1,1),activation='relu', padding= 'same')(i)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(64, (3,3), strides = (1,1),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.2)(x)


x = layers.Conv2D(128, (3,3), strides = (1,1),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (3,3), strides = (1,1),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)


x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(3, activation='softmax')(x) 


model_cnn = Model(i,x)
opt=Adam(learning_rate=0.0001)
model_cnn.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#create callbacks for early stopping and restoring best weights
callBack = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model_cnn.fit(training_set,validation_data=test_set,epochs=3, callbacks=[callBack])   


preds_cnn = model_cnn.predict(test_set)
preds_cnn = preds_cnn.argmax(1)
cm = confusion_matrix(test_set.labels, preds_cnn)
plot_confusion_matrix(cm, class_names)


######################################################################################
#ResNet152V2
######################################################################################


foundationModel = ResNet152V2(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in foundationModel.layers:
  layer.trainable = False

headModel = foundationModel.output
x = layers.MaxPooling2D((2, 2))(headModel)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
#x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(3, activation='softmax')(x)


model_resnet = Model(inputs=foundationModel.input, outputs = x)



opt=Adam(learning_rate=0.0001)
model_resnet.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#create callbacks for early stopping and restoring best weights
callBack = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model_resnet.fit(training_set,validation_data=test_set,epochs=4, callbacks=[callBack])   

preds_resnet = model_resnet.predict(test_set)
preds_resnet = preds_resnet.argmax(1)
cm = confusion_matrix(test_set.labels, preds_resnet)
plot_confusion_matrix(cm, class_names)


wrongOnes_resnet = np.where(test_set.labels != preds_resnet)

wrongPredPath_resnet = []
for location in wrongOnes_resnet[0]:
   spot = test_set.filepaths[location]
   wrongPredPath_resnet.append(spot)

rightOnes_resnet = np.where(test_set.labels == preds)

rightOnesPath_resnet = []
for location in rightOnes_resnet[0]:
   spot = test_set.filepaths[location]
   rightOnesPath_resnet.append(spot)

######################################################################################
#mobilenet_v3
######################################################################################

foundationModel = MobileNetV2(input_shape=(224,224,3), weights='imagenet', include_top=False)

for layer in foundationModel.layers:
  layer.trainable = False

headModel = foundationModel.output
x = layers.MaxPooling2D((2, 2))(headModel)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(128, (16,16), strides = (3,3),activation='relu', padding= 'same')(x)
x = layers.BatchNormalization()(x)
#x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.3)(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(3, activation='softmax')(x)


model_mobilnet= Model(inputs=foundationModel.input, outputs = x)


opt=Adam(learning_rate=0.0001)
model_mobilnet.compile(optimizer=opt,loss='categorical_crossentropy',metrics=['accuracy'])

#create callbacks for early stopping and restoring best weights
callBack = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', min_delta=0, patience=2, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

history = model_mobilnet.fit(training_set,validation_data=test_set,epochs=5, callbacks=[callBack])   


preds_mobilnet = model_mobilnet.predict(test_set)
preds_mobilnet = preds_mobilnet.argmax(1)
cm = confusion_matrix(test_set.labels, preds_mobilnet)
plot_confusion_matrix(cm, class_names)



wrongOnes_mobilnet = np.where(test_set.labels != preds_mobilnet)

wrongPredPath_mobilnet = []
for location in wrongOnes_mobilnet[0]:
   spot = test_set.filepaths[location]
   wrongPredPath_mobilnet.append(spot)

rightOnes_mobilnet = np.where(test_set.labels == preds)

rightOnesPath_mobilnet = []
for location in rightOnes_mobilnet[0]:
   spot = test_set.filepaths[location]
   rightOnesPath_mobilnet.append(spot)


#######################################################################################

#get the paths and plot
file_path =  '/test/COVID19/COVID19(477).jpg'
test_image = cv2.imread(data_path + file_path)
test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
plt.title('Covid')
plt.imshow(test_image)

file_path =  '/test/NORMAL/NORMAL(1284).jpg'
test_image = cv2.imread(data_path + file_path)
test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
plt.title('Normal')
plt.imshow(test_image)

file_path =  '/test/PNEUMONIA/PNEUMONIA(3439).jpg'
test_image = cv2.imread(data_path + file_path)
test_image = cv2.resize(test_image, (224,224),interpolation=cv2.INTER_NEAREST)
plt.title('Pneumonia')
plt.imshow(test_image)

#get accuracy scores
accuracy_score(test_set.labels, preds_ann)
accuracy_score(test_set.labels, preds)
accuracy_score(test_set.labels, preds_resnet)
accuracy_score(test_set.labels, preds_mobilnet)

#plot some limes 
x,y,z = sample(rightOnesPath,3)
fig,ax = plt.subplots(1,3,figsize=(18,6))
visualize(model_ann, x,ax[0],0.15)
visualize(model, x,ax[1],0.4)
visualize(model_mobilnet, x,ax[2],0.7)



