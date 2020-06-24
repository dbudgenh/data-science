# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 21:19:33 2020

@author: david
"""
import os
from imutils import paths
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set() #set default theme for the graphs

#Loads the data from the directories ('normal','covid','pnemonia')
#Saves the path of images in an array and the labels in a seperate array


DATA_DIRECTORY = 'data'
DIRECTORIES = ['normal','corona','pnemonia']
LABELS_DICT= dict(zip(DIRECTORIES,range(len(DIRECTORIES))))
INVERSE_LABELS_DICT = {v: k for k, v in LABELS_DICT.items()}

def create_metadata(df):
    #new_df = df.copy()
    #new_df['label'] = new_df['label'].apply(lambda x: INVERSE_LABELS_DICT[x])
    
    df_new_columns = pd.DataFrame(df['filename'].str.split(os.sep).tolist(),
                                 columns = ['mainfolder','subfolder','filename'])
    
    result_df = df_new_columns.join(df['label'])
    result_df.to_csv('merged_metadata.csv',index=False,encoding='utf-8')

def resample_df(df,amount_of_samples):
    df_normal = df[df.label == 0]
    df_pneomnia = df[df.label == 1]
    df_corona = df[df.label == 2]
    #print('Normal images: {}  \nPneomnia images: {} \nCorona images: {}'.format(len(df_normal),len(df_pneomnia),len(df_corona)))
    df_normal = df_normal.sample(amount_of_samples)
    df_pneomnia = df_pneomnia.sample(amount_of_samples)
    df_corona = df_corona.sample(amount_of_samples)
    #print('Normal images: {}  \nPneomnia images: {} \nCorona images: {}'.format(len(df_normal),len(df_pneomnia),len(df_corona)))
    df_resampled = df_normal.append(df_pneomnia).append(df_corona)
    return df_resampled
    
def plot_image_count(df):
    new_df = df.copy()
    new_df['label'] = new_df['label'].apply(lambda x: INVERSE_LABELS_DICT[x])
    header = 'label'
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    sns.countplot(x=header, data=new_df)
    plt.show()

def load_data():
    df = pd.DataFrame()
    data = []
    labels = []
    
    for directory in DIRECTORIES:
        data_directory = os.path.join(DATA_DIRECTORY,directory)
        images = list(paths.list_images(data_directory))
        for image in images:
            data.append(image)
            labels.append(LABELS_DICT[directory])
            
    df['filename'] = data
    df['label'] = labels
    return df
    #return np.array(images),np.array(labels)

def main():
    #The data consists of x-ray images, each image is annotated with either 'normal','pneumonia (no covid)' or 'covid'
    #Loads the data from the directories 'normal','corona','pnemonia' and saves them into a dataframe
    df = load_data()
   
    #Creates the metadata for the complete merged dataset
    create_metadata(df)
    
    print('Initial Dataframe')
    print(df.head(5))
    #                                                 filename  label
    # 0  data\normal\F051E018-DAD1-4506-AD43-BE4CA29E96...      0
    # 1                      data\normal\IM-0001-0001.jpeg      0
    # 2                      data\normal\IM-0003-0001.jpeg      0
    # 3                      data\normal\IM-0005-0001.jpeg      0
    # 4                      data\normal\IM-0006-0001.jpeg      0
    
    #graph of the distributions
    plot_image_count(df)
    
    #After inspeccting the distributions, its obvious that the dataset needs to be resampled
    #Creates a new dataframe, with a fixed amount of samples for each label
    print('Resampled Dataframe')
    df_resampled = resample_df(df,amount_of_samples=175)
    print(df_resampled.sample(5))
    
    #print(len(df_resampled))

    
    
    
    
    
    
    
    #We can see that the dataset is imbalanced for the classes
    #There are 1576 'normal' xray images
    #There are 175 'covid19' xray images
    #There are 4276 'pneunomia' xray images
    
    #We rebalance the dataset first.
    
    #X = images
    #convert to one-hot-encoded
    #Y = to_categorical(labels,3)
    
    #print(len(X))
    #print(len(Y))

if __name__ == "__main__":
    main()