# Data Science Project
For a detailed project report, click [here](https://docs.google.com/document/d/1aLG9sdlID6obtIDXQdwd9rh348x2RW1xFRRMtF4gU7w/edit).

## First Steps
1. Download the [first dataset](https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset) and the [second dataset](https://www.kaggle.com/bachrr/covid-chest-xray)
1. After the download is complete, extract both datasets in the directory 'data'

## Executing the code
1. In the main directory, run the command:
```console
python load_data.py
```
This will extract all the images from the first dataset and move the images to the directories 'normal','corona','pneumonia', based on their labels.
The directories can be found in in the directory 'data'.

2. Run the command:
```console
python load_data2.py
```
This will extract all the images from the second dataset, in the same manner as for the first.

3. Now run the command:
```console
python train.py
```
This will use the merged datasets, to train a neural network and print it's metrics. The weights/model/history will be saved in the corresponding directory.

## Alternative
1. Extract the file 'dataset.zip' in the directory 'data', this is the merged dataset. The file 'merged_metadata' is the corresponding metadata file.
1. Now run the command:
```console
python train.py
```
