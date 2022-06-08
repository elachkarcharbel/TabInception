# TabInception

### Leveraging Computer Vision Techniques for Guitar Tablature Transcription


### 0. Requirements

This project was made to be run with Python 3.9. You should have the following libraries/packages installed:
* numpy
* scipy
* pandas
* jams
* librosa
* keras
* tensorflow

```pip install -r requirements.txt```

### 1. Download dataset

Download the GuitarSet audio and annotation data from [here](https://zenodo.org/record/1422265/files/GuitarSet_audio_and_annotation.zip?download=1 "GuitarSet download").

Unzip and place the downloaded GuitarSet folder in `TabInception/data/` so that in `TabInception/data/GuitarSet/` you have the following two folders:
* `annotation/`
* `audio/`

The remaining instructions assume that you are in the `TabInception/` folder.

### 2. Preprocessing

#### Compute the Constant-Q Transform on the audio files
Run the following line to preprocess different spectral representations for the audio files: 

  `bash data/Bash_TabDataReprGen.sh`

This will save the preprocessed data as compressed numpy (.npz) files in the `data/spec_repr/` directory.

#### Convert the NPZ files to tfrecords

Run either the `np_to_tfrecords_192x9.py` or `np_to_tfrecords_224x224.py` to slice your computed CQT features into images of size 192x9. The output of the first code would be the images of size 192x9 with their corresponding labels. The output of the second one would be the images of size 224x224 with their corresponding labels.

### 3. Training

In this repository, we offer an implementation of the EfficientNetB0, the SwinTransformer,the Vision Transformer, and the TabInception model in tensorflow for automatic guitar transcription using the GuitarSet dataset.

Run the following command to train the TabInception model:

`python model/TabInception_tfrec_192x9.py`

Run the following command to train the Vision Transformer model:

`python model/VIT_tfrec_192x9.py`

Run the following command to train the EfficientNetB0 model:

`python model/EfficientNetB0_tfrec_224x224.py`

Run the following command to train the SwinTransformer model:

`python model/Swin_tfrec_224x224.py`


### 4. Evalutation

A summary log and a csv results file will be saved in a time-stamped folder within the `model/saved/` directory. Additionally, a folder for each fold of data will be created, containing the individual model's weights and predictions. 


