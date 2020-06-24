# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:58:08 2020

@author: david
"""

from data_process import load_data
from data_process import resample_df
import Augmentor
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import  Input, Conv2D, MaxPooling2D,GlobalMaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, Activation, MaxPool2D, AvgPool2D, Dropout,BatchNormalization
from traininggenerator import TrainingGenerator
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import efficientnet.tfkeras as efn
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


#CONSTANTS
VALIDATION_RATIO = 0.15
IMG_SIZE = 260
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE =1e-3
CHECKPOINT_DIRECTORY = 'weights'
TENSORBOARD_DIRECTORY = 'tensorboard'
HISTORY_DIRECTORY = 'history'
MODEL_DIRECTORY = 'model'
CONFUSION_MATRIX_DIRECTIORY = 'cmatrix'
USE_DATA_AUGMENTATION = True


#Returns the model, either naive or efficientnet
def getModel(modelName,img_size):
    def getNaiveModel():
        visible = Input(shape=(img_size, img_size, 1), dtype=tf.float32)
        conv = Conv2D(32, (5, 5))(visible)
        conv_act = Activation('relu')(conv)
        conv_act_batch = BatchNormalization()(conv_act)
        conv_maxpool = MaxPooling2D()(conv_act_batch)
        conv_dropout = Dropout(0.1)(conv_maxpool)
    
        conv = Conv2D(64, (3, 3))(conv_dropout)
        conv_act = Activation('relu')(conv)
        conv_act_batch = BatchNormalization()(conv_act)
        conv_maxpool = MaxPooling2D()(conv_act_batch)
        conv_dropout = Dropout(0.2)(conv_maxpool)
    
        conv = Conv2D(128, (3, 3))(conv_dropout)
        conv_act = Activation('relu')(conv)
        conv_act_batch = BatchNormalization()(conv_act)
        conv_maxpool = MaxPooling2D()(conv_act_batch)
        conv_dropout = Dropout(0.3)(conv_maxpool)
    
        conv = Conv2D(256, (3, 3))(conv_dropout)
        conv_act = Activation('relu')(conv)
        conv_act_batch = BatchNormalization()(conv_act)
        conv_maxpool = MaxPooling2D()(conv_act_batch)
        conv_dropout = Dropout(0.4)(conv_maxpool)
    
        conv = Conv2D(512, (3, 3))(conv_dropout)
        conv_act = Activation('relu')(conv)
        conv_act_batch = BatchNormalization()(conv_act)
        conv_maxpool = MaxPooling2D()(conv_act_batch)
        conv_dropout = Dropout(0.5)(conv_maxpool)
        
        gap2d = GlobalAveragePooling2D()(conv_dropout)
        act = Activation('relu')(gap2d)
        batch = BatchNormalization()(act)
        dropout = Dropout(0.3)(batch)
    
        fc1 = Dense(256)(dropout)
        act = Activation('relu')(fc1)
        batch = BatchNormalization()(act)
        dropout = Dropout(0.4)(batch)
    
        # and a logistic layer
        predictions = Dense(3, activation='softmax',name='prediction')(dropout)
        
        # Create model.
        model = tf.keras.Model(visible, predictions, name=modelName)
        return model
    
    def getEfficientNetModel():
        model = efn.EfficientNetB2(include_top=True,
                                         weights=None,
                                         input_shape=(img_size,img_size,1),
                                         classes=3)
        return model
    
    if modelName == 'naive':
        return getNaiveModel(),modelName
    elif modelName == 'efficientNet':
        return getEfficientNetModel(),modelName
    else:
        raise Exception("Wrong model name! Either use 'naive' or 'efficientNet' as model name!")

#Data Augmentation operations
def getAugmentorPipeline():
        p = Augmentor.Pipeline()
        #stack operations...
        p.zoom(probability=0.25,min_factor=0.75,max_factor=1.25)
        p.random_brightness(probability=0.25,min_factor=0.75,max_factor=1.25)
        p.flip_left_right(probability=0.25)
        p.rotate(probability=0.25,max_left_rotation=10,max_right_rotation=10)
        p.shear(probability=0.25, max_shear_left=10, max_shear_right=10)
        p.random_erasing(probability=0.25,rectangle_area=0.25)
        return p
    
#Used for the learning rate schedular
def schedular(epoch):
    #for the first quarter of epochs, use the initial learning rate
    if epoch < EPOCHS * 0.25:
        return LEARNING_RATE
    #after the first quarter of epochs and before half of the epochs, multiply the learning rate by 0.1
    elif epoch < EPOCHS *0.5:
        return LEARNING_RATE*0.2
    #third quarter, multiply by 0.01
    elif epoch < EPOCHS * 0.75:
        return LEARNING_RATE*0.04
    #last quarter, multiply by 0.001
    return LEARNING_RATE*0.008

def splitData(X,Y,validation_ratio=VALIDATION_RATIO):
    data_size = len(X)
    indices = np.arange(data_size)
    #shuffle in place
    np.random.shuffle(indices)
    #shuffle input and output
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    
    train_size = int(data_size * (1-validation_ratio))
    
    #slice the shuffled input
    X_train,X_test = X_shuffled[:train_size],X_shuffled[train_size:]
    #slice the shuffled output
    Y_train,Y_test = Y_shuffled[:train_size],Y_shuffled[train_size:]
    
    assert len(X_train) + len(X_test) == data_size
    assert len(X_train) == len(Y_train)
    assert len(X_test) == len(Y_test)
    
    return X_train,X_test,Y_train,Y_test

def main():
    #Load the merged dataset, return a dataframe
    df = load_data()
    #Resample the dataframe, to balance the distribution between the classes
    resampled_df = resample_df(df,amount_of_samples=175)
    #X are the X-ray images of the patients
    X = resampled_df['filename'].to_numpy()
    #Y are the labels corresponding to each patient, either 0 ('normal'), 1 ('corona') or 2 ('pnemonia')
    Y = resampled_df['label'].to_numpy()
    #Convert to one-hot-encoding (simplifies training process), label 0 becomes [1,0,0], 1 becomes [0,1,0], 2 becomes [0,0,1]
    Y = to_categorical(Y,3) #one-hot encoding
    #Split the data into training- and test set
    X_train,X_test,Y_train,Y_test = splitData(X,Y,validation_ratio=VALIDATION_RATIO)
    #get our Neural network model, for now: either a naive model, or state-of-the-art Efficient-Net model
    
    model,modelName = getModel('naive',IMG_SIZE)
    #print out info about the models(layer structures etc)
    model.summary()
    #Adam is an optimization algorithm that can be used instead of the classical stochastic gradient descent procedure to update network weights iterative based in training data
    #Most commonly used, can be changed out by SGD or anything similar
    optimizer = Adam(learning_rate=LEARNING_RATE)
    
    #best to use categorial loss for classification problems
    loss_fuction = 'categorical_crossentropy'
    loss=[loss_fuction]
    
    #Metrics to measure during training, we are only interested in the prediction accuracy for now
    metrics = {'probs':'accuracy'}
    #set the optimizer, loss function, and the metrics to print when training
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics= metrics)
    
    #List of Utilities called at certain points during model training
    callbacks = [LearningRateScheduler(schedular), #schedule the learning rate
                 ModelCheckpoint(os.path.join(CHECKPOINT_DIRECTORY,modelName + '{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='val_loss', #monitor validation loss
                                 verbose=1, #print fancy progressbar
                                 save_best_only=True, #self explanatory
                                 mode='auto', #the decision to overwrite current save file
                                 save_weights_only=True, #save only the weights, not full model
                                 save_freq = 'epoch' ), #save after every epoch
                 TensorBoard(log_dir=TENSORBOARD_DIRECTORY,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)]
                 
        
    p = getAugmentorPipeline()

    training_generator = TrainingGenerator(augmentor_pipeline=p,
                                                images_filename=X_train,
                                                labels=Y_train,
                                                batch_size = BATCH_SIZE,
                                                img_size=IMG_SIZE,
                                                normalize=True,
                                                data_aug=True if USE_DATA_AUGMENTATION else False)
         
    validation_generator = TrainingGenerator(augmentor_pipeline=p,
                                                images_filename=X_test,
                                                labels=Y_test,
                                                batch_size=BATCH_SIZE,
                                                img_size=IMG_SIZE,
                                                normalize=True,
                                                data_aug=False)
    print('Training model...')
    history = model.fit_generator(generator=training_generator,
                            steps_per_epoch = len(X_train) // BATCH_SIZE,
                            validation_data = validation_generator,
                            validation_steps =len(X_test) // BATCH_SIZE,
                            epochs=EPOCHS,
                            verbose=1,
                            callbacks=callbacks,
                            workers=6,
                            use_multiprocessing = False,
                            shuffle = True,
                            initial_epoch=0,
                            max_queue_size =6
                            )
    #Confution Matrix and Classification Report
    Y_pred = model.predict_generator(validation_generator, len(X_test) // BATCH_SIZE+1)
    y_pred = np.argmax(Y_pred, axis=1)
    
    print('Saving confusion matrix as {}.png'.format(modelName))
    target_names = ['normal','corona','pnemonia']
    c_matrix = confusion_matrix(validation_generator.get_all_labels(), y_pred)
    df_cm = pd.DataFrame(c_matrix, index = [i for i in target_names],
                  columns = [i for i in target_names])
    plt.figure(figsize = (10,10))
    plot = sns.heatmap(df_cm, annot=True)
    plot.figure.savefig(os.path.join(CONFUSION_MATRIX_DIRECTIORY,'{}.png'.format(modelName)))
    
    print('Classification Report')
    print(classification_report(validation_generator.get_all_labels(), y_pred, target_names=target_names))
    
    
    
    print('Saving history and model...')
    #Save the history 
    with open(os.path.join(HISTORY_DIRECTORY,modelName + '.h5'),'wb') as f:
        pickle.dump(history.history,f)
        
    #Save the whole model
    model.save(os.path.join(MODEL_DIRECTORY,modelName + '.h5'))

if __name__ == "__main__":
    main()