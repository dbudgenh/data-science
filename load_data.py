# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 21:46:46 2020

@author: david
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import math
import shutil

sns.set_style('dark') #set theme for the graphs

#Directory where all datasets are extracted into
DATA_DIRECTORY = 'data'
#Directory of the extracted dataset
MAIN_DIRECTORY = os.path.join(DATA_DIRECTORY,'Coronahack-Chest-XRay-Dataset')
IMAGE_DIRECOTRY = os.path.join(MAIN_DIRECTORY,'Coronahack-Chest-XRay-Dataset')
#Used for plotting graphs
PLOT_DIRECTORY = 'plot'

DATASET_CSV = os.path.join(MAIN_DIRECTORY,'Chest_xray_Corona_Metadata.csv')
SUMMARY_CSV = os.path.join(MAIN_DIRECTORY,'Chest_xray_Corona_dataset_Summary.csv')

#Plot an image, given the path to file
def plot_img(path_to_img,title=''):
    im = cv2.imread(path_to_img)
    plt.axis('off')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),cmap='gray')
    plt.title(title)
    #plt.savefig(os.path.join(PLOT_DIRECTORY,'{}.png'.format(filename)))
    plt.show()
    
#Plot an image, given an numpy array 
def plot_array(arr):
    plt.axis('off')
    plt.imshow(arr,cmap='gray')
    plt.show()
    
#Function called for every row of a dataframe, used to plot the images of a dataframe
def plot_row(row):
    dataset_type,filename,label,virus_type= row['Dataset_type'],row['X_ray_image_name'],row['Label'],row['Label_2_Virus_category']
    title = 'Label: {} \n Virus: {}'.format(label,virus_type)
    plot_img(os.path.join(IMAGE_DIRECOTRY,dataset_type,filename),title)

#Saves images that are normal in the 'normal' folder
#Saves images that have pnemonia (and no covid) in the 'pnemonia' folder
#Saves images that have covid (pnemonia + covid) in the 'corona' folder
def move_to_destination_folder(row):
    dataset_type,filename,label,virus_type= row['Dataset_type'],row['X_ray_image_name'],row['Label'],row['Label_2_Virus_category']
    has_covid = label == 'Pnemonia' and virus_type == 'COVID-19'
    source_file_path = os.path.join(IMAGE_DIRECOTRY,dataset_type,filename)
    destination_file_path = os.path.join(DATA_DIRECTORY,'corona' if has_covid else label,filename)
    shutil.copy2(source_file_path,destination_file_path)

def main():
    dataset_df = pd.read_csv(DATASET_CSV)
    valid_df = pd.read_csv(SUMMARY_CSV)
    
    #Print information about the dataset
    print(valid_df.to_string())
      # Unnamed: 0     Label Label_1_Virus_category Label_2_Virus_category  Image_Count
# 0           0    Normal                    NaN                    NaN         1576
# 1           1  Pnemonia         Stress-Smoking                   ARDS            2
# 2           2  Pnemonia                  Virus                    NaN         1493
# 3           3  Pnemonia                  Virus               COVID-19           58
# 4           4  Pnemonia                  Virus                   SARS            4
# 5           5  Pnemonia               bacteria                    NaN         2772
# 6           6  Pnemonia               bacteria          Streptococcus            5


    
    #show 5 first rows of the dataset
    print(dataset_df.head(5).to_string())
#     Unnamed: 0   X_ray_image_name   Label Dataset_type Label_2_Virus_category Label_1_Virus_category
# 0           0  IM-0128-0001.jpeg  Normal        TRAIN                    NaN                    NaN
# 1           1  IM-0127-0001.jpeg  Normal        TRAIN                    NaN                    NaN
# 2           2  IM-0125-0001.jpeg  Normal        TRAIN                    NaN                    NaN
# 3           3  IM-0122-0001.jpeg  Normal        TRAIN                    NaN                    NaN
# 4           4  IM-0119-0001.jpeg  Normal        TRAIN                    NaN                    NaN
    
    
    headers = ['Label','Label_2_Virus_category']
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    #Shows the amount of normal patients and infected patients (Pnemonia)
    first_plot = sns.countplot(x=headers[0], data=dataset_df, ax=ax[0])
    #Shows the distribution of the Pnemonia infected patients (omitted NaN values for clarity)
    second_plot = sns.countplot(x=headers[1], data=dataset_df, ax=ax[1])
    plt.show()
    
    #first_plot.figure.savefig(os.path.join(PLOT_DIRECTORY,'{}.png'.format('distribution_first_dataset')))
    
    #That means we have XRay values for normal patients (not infected) or patients with Pnemonia.
    #Only a subsets of these pneomnia patients have the virus COVID-19.
    
    #Plot some samples
    sample_df = dataset_df.sample(5)
    sample_df.apply(plot_row,axis=1)
    
    
    #Move all the files to the correct folder, this is done so we can merge multiple datasets later
    dataset_df.apply(move_to_destination_folder,axis=1)
  
    
    

if __name__ == "__main__":
    main()
