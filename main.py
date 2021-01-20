from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout
from PyQt5.QtWidgets import QPushButton, QFileDialog , QLabel, QLineEdit
from imutils import paths
import numpy as np
import argparse
import os
import sys
import cv2
from PyQt5.QtCore import Qt
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 
import random
import cv2
import time
import scipy.misc
from pydub import AudioSegment
from pydub.silence import split_on_silence
from utils import mfcc_feature_extraction, convert_mfcc2img
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
from utils import read_conf



class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.title = "VOICENET"
        self.top = 200
        self.left = 0
        self.width = 750
        self.height = 500
        self.mfcc = None
        self.options = read_conf()
        self.model = load_model(self.options.model_path)
        self.img_link = np.load(self.options.img_link, allow_pickle=True).item()
        self.name_link = np.load(self.options.name_link, allow_pickle=True).item()
        self.InitWindow()
        

    def InitWindow(self):
        self.setWindowIcon(QtGui.QIcon("icon/icon.png"))
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        vbox = QVBoxLayout()

        #button load data
        self.btn_loaddata = QtWidgets.QPushButton('Load Data')
        self.btn_loaddata.setGeometry(QtCore.QRect(20, 190, 75, 23))
        self.btn_loaddata.clicked.connect(self.load_data)
        vbox.addWidget(self.btn_loaddata)


        self.lb_img = QtWidgets.QLabel('Load Image')
        self.lb_img.setGeometry(QtCore.QRect(10, 30, 531, 131))
        self.lb_img.setMinimumSize(QtCore.QSize(0, 0))
        self.lb_img.setText("")
        vbox.addWidget(self.lb_img)
 
        #button predict
        self.btn_run = QtWidgets.QPushButton('Predict')
        self.btn_run.setGeometry(QtCore.QRect(140, 190, 75, 23))
        self.btn_run.setMouseTracking(False)
        self.btn_run.setAcceptDrops(False)
        self.btn_run.setAutoFillBackground(False)
        self.btn_run.clicked.connect(self.predict)
        vbox.addWidget(self.btn_run)

        self.lb_result = QtWidgets.QLabel('Result')
        self.lb_result.setGeometry(QtCore.QRect(10, 20, 191, 141))
        self.lb_result.setText("")
        self.lb_result.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.lb_result)

        self.setLayout(vbox)

        self.show()


    def load_data(self):
        #opening dialog
        fname = QFileDialog.getOpenFileName(self, 'Open File', self.options.open_in, "Wav Files (*.wav)")

        #get file name
        wavPath = fname[0]
        file_name = wavPath.split('/')[-1].split('.')[0]
        
        #read wav file
        signal = AudioSegment.from_wav(wavPath)
        
        
        #reduce silence
        N = len(signal)
        
        if N > 3000: #(3s)
            ranges = split_on_silence(signal, min_silence_len=200, keep_silence=20, silence_thresh=-30)
            wav = AudioSegment.empty()
            for item in ranges:
                wav += item
            signal = wav
        
        #update len signal
        N = len(signal)
        
        if N > 2000:
            signal = signal[0:2000]
    
        signal.export(self.options.img_raw, format='wav')

        #get a random image
        k = random.randint(1,3)
        
        if k == 1:
            imagePath = self.options.img_1
        elif k == 2:
            imagePath = self.options.img_2
        else:
            imagePath = self.options.img_3
        
        if imagePath is not None:
            pixmap = QPixmap(imagePath)
            self.lb_img.setPixmap(QPixmap(pixmap))
        else:
            return 0;
    

    def predict(self):
        #read raw file
        mfcc = mfcc_feature_extraction(self.options.img_raw)
        mfcc = convert_mfcc2img(mfcc)
        
        #convert to tensor
        mfcc = mfcc.reshape(1, 40, 80, 1)
        predicted_vector = self.model.predict(mfcc)
        dic = dict(zip(np.arange(0,len(predicted_vector[0])), predicted_vector[0]))
        sorted_dic = {k:v for k, v in sorted(dic.items(), key=lambda x: x[1])}

        #get keys and values
        keys=[i for i in sorted_dic.keys()]
        values=[j for j in sorted_dic.values()]

        #set config matplotlib.pyplot
        fig = plt.figure(figsize=(7, 2)); ax = []
        plt.rcParams['figure.facecolor'] = '#262D37'
        plt.rcParams['savefig.facecolor']='#262D37'

        for i in range(3):
            img = imread(self.img_link[keys[-i]])
            #create subplot and append to ax
            ax.append( fig.add_subplot(1, 3, i+1) )
            ax[-1].set_title(self.name_link[keys[-i]] + '-' + str(round(values[-i]*100,2))+'%', color='white')  # set title
            #hiden
            ax[-1].get_xaxis().set_ticks([])
            ax[-1].axes.get_yaxis().set_ticks([])
            plt.imshow(img)
      
        plt.savefig(self.options.img_result)

        
        pixmap = QPixmap(self.options.img_result)
        self.lb_result.setPixmap(QPixmap(pixmap))
        pass;

if __name__ == "__main__":

    App = QApplication(sys.argv)

    style = """
        QWidget{
            background: #262D37;
        }

        QLabel{
            color: #fff;
        }

        QLabel#round_count_label, QLabel#highscore_count_label{
            border: 1px solid #fff;
            border-radius: 8px;
            padding: 2px;
        }
        QPushButton
        {
            color: white;
            background: #0577a8;
            border: 1px #DADADA solid;
            padding: 5px 10px;
            border-radius: 2px;
            font-weight: bold;
            font-size: 9pt;
            outline: none;
        }

        QPushButton:hover{
            border: 1px #C6C6C6 solid;
            color: #fff;
            background: #0892D0;
        }

        QLineEdit {
            padding: 1px;
            color: #fff;
            border-style: solid;
            border: 2px solid #fff;
            border-radius: 8px;
        }

    """
    App.setStyleSheet(style)
    
    window = Window()
    sys.exit(App.exec())