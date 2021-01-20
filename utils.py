import numpy as np
from imutils.paths import list_files
import os
import librosa
import cv2
from imutils.paths import list_images
import configparser as ConfigParser
from optparse import OptionParser


def mfcc_feature_extraction(file_name):
    """MFCC Feature Extraction

    Args:
        file_name (string): wav file name

    Returns:
        [2D array]: MFCC features
    """
    max_pad_len = 100
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered: ", e)
        return None 
     
    return mfccs


def convert_mfcc2img(mfcc):
    """Convert MFCC features to image

    Args:
        mfcc (2D array): mfcc features

    Returns:
        [2D array]: Image corresponding to MFCC features
    """
    try:
      mfcc = np.array(mfcc)
      
      MAX=np.max(mfcc);MIN=np.min(mfcc)
      
      #new value domain
      NEW_MAX=255; NEW_MIN=0
      
      mfcc = (mfcc-MIN)/(MAX-MIN) * (NEW_MAX-NEW_MIN)

      mfcc = mfcc[:,0:80]

      mfcc = np.round(mfcc)

    except Exception as e:
      print("Error encountered: ", e)
      return None;

    return mfcc;


def read_conf():
    """Read config file

    Returns:
        [Namespace object]: containing the arguments to the command
    """
    parser=OptionParser()
    parser.add_option("--cfg") # Mandatory
    (options,args)=parser.parse_args()
    cfg_file=options.cfg
    Config = ConfigParser.ConfigParser()
    Config.read(cfg_file)
    
    #[image]
    options.img_1 = Config.get('image', 'img_1')
    options.img_2 = Config.get('image', 'img_2')
    options.img_3 = Config.get('image', 'img_3')
    options.img_raw = Config.get('image', 'img_raw')
    options.img_result = Config.get('image', 'img_result')
    
    #[data]
    options.open_in = Config.get('data', 'open_in')
    
    #[model]
    options.model_path = Config.get('model', 'model_path')
    
    #[link]
    options.img_link = Config.get('link', 'img_link')
    options.name_link = Config.get('link', 'name_link')
    
    return options