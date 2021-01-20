
# Genetic VoiceNet [![CircleCI](https://circleci.com/gh/faustomorales/keras-ocr.svg?style=shield)](https://github.com/nguyentruonglau) [![Documentation Status](https://readthedocs.org/projects/keras-ocr/badge/?version=latest)](https://github.com/nguyentruonglau)

Application of VoiceNet - convolutional neural network architecture found by nondominated sorting genetic algorithm ii with code: 0-10 - 1-01-001 -  0-00.
Two goals are optimized: accuracy and cost calculation
> [Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition]
>
> Nguyễn Trường Lâu - Student at University of Information Technology (UIT)
>

![overview](https://github.com/nguyentruonglau/voicenet-app/blob/main/img/gui.png "VoiceNet App")


## Requirements
``` 
ffmpeg-python==0.2.0
ffprobe-python==1.0.3
h5py==2.10.0
imutils==0.5.4
Keras==2.3.1
librosa==0.8.0
matplotlib==3.3.3
numpy==1.19.5
opencv-contrib-python==4.5.1.48
Pillow==8.1.0
pydub==0.24.1
PyQt5==5.15.2
scikit-learn==0.24.1
scipy==1.6.0
tensorflow==1.15.2

pip install -r requirements.txt
```

## Pretrained models
``` 
model/voicenet.hdf5
```

## Run app
``` 
python main.py --cfg=cfg/config.cfg
```

## Citations
If you find the code useful for your research, please consider citing our works
``` 
@article{voicenet,
  title={Evolutionary Neural Architecture Search For Vietnamese Speaker Recognition},
  author={Nguyễn Trường Lâu - Student at University of Information Technology (UIT)},
  booktitle={NAS},
  year={2020}
}
```
