# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 19:41:44 2020

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
#sns.set_style('dark') #set theme for the graphs

from load_data import plot_img
from load_data import PLOT_DIRECTORY

#Directory where all datasets are extracted into
DATA_DIRECTORY = 'data'
IMAGE_DIRECOTRY = os.path.join(DATA_DIRECTORY,'images')
DATASET_CSV = os.path.join(DATA_DIRECTORY,'metadata.csv')

def plot_row(row):
    virus_type,view,filename = row['finding'],row['view'],row['filename']
    title = 'Virus: {}'.format(virus_type)
    plot_img(os.path.join(IMAGE_DIRECOTRY,filename),title)

def move_to_destination_folder(row):
    virus_type,filename = row['finding'], row['filename']
    source_file_path = os.path.join(IMAGE_DIRECOTRY,filename)
    has_covid = 'COVID-19' in virus_type
    
    #We only extract the covid-19 images (to increase the sample size)
    if not has_covid:
        return
    
    #some files in the metadata contain invalid paths..
    if not os.path.exists(source_file_path):
        return
   

    destination_file_path = os.path.join(DATA_DIRECTORY,'corona',filename)
    shutil.copy2(source_file_path,destination_file_path)

def main():
    dataset_df = pd.read_csv(DATASET_CSV)
    #We only inspect images with posteroanterior (PA) views
    
    headers = ['finding']
    fig, ax = plt.subplots(1, 1, figsize=(300, 300))
    #Shows the distribution of findings
    first_plot = sns.countplot(x=headers[0], data=dataset_df)
    first_plot.set_xticklabels(first_plot.get_xticklabels(), fontsize=8,rotation=22)
    first_plot.figure.savefig(os.path.join(PLOT_DIRECTORY,'{}.png'.format('distribution_second_dataset')))
    
    plt.show()
    
    
    dataset_df = dataset_df[(dataset_df.view == 'PA')]
    #Example images
    sample_df = dataset_df.sample(3)
    sample_df.apply(plot_row,axis=1)
    
    dataset_df.apply(move_to_destination_folder,axis=1)
    
    #print(dataset_df.head(1).to_string())
if __name__ == "__main__":
    main()