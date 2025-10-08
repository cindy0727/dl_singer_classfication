# dl singer identification

### Usage
#### extract_fea.py
Extracting melspectrograms of artist20 
1. change the path of the folder in extract_fea.py line 73, 76, 77, 83, 85, 91 and 93
```
73: ROOT     = # path of artist20
76: art_dir  = # path of training data and validation data
77: save_dir = # data saving path / 'song_data_artist20_origin'

# 以下照常來說是存取純人聲和純背景音樂的，雖然我只有做原始音檔分析，但還是需要這些路徑才能跑程式
83: art_dir  = # path of training data and validation data
85: save_dir = # data saving path / 'song_data_artist20_origin'
91: art_dir  = # path of training data and validation data
93: save_dir = # data saving path / 'song_data_artist20_origin'
```
2. in terminal
```
pip install requirements.txt
python extract_fea.py
```
#### train_CRNN.py
```
usage: train_CRNN.py [-h] [-class CLASSES_NUM] [-gid GPU_INDEX]
                     [-bs BATCH_SIZE] [-lr LEARN_RATE] [-val VAL_NUM]  
                     [-stop STOP_NUM] [-rs RANDOM_STATE] [--origin] [--vocal]
                     [--remix] [--all] [--CRNNx2] [--debug]

optional arguments:
  -class, classes number (default:20)
  -gid, gpu index (default:0)
  -bs, batch size (default:100)
  -lr, learn rate (default:0.0001)
  -val, valid per epoch (default:1)
  -stop, early stop (default:20)
  -rs random state (default:0)
  --origin, use original audio to training
  --vocal, use separated vocal audio to training
  --remix, use remix audio to training
  --all, use all of the above data to training
  --CRNNx2, use CRNNx2 model to training
  --debug, debug mode
```
#### predict_on_audio.py
```
python predict_on_audio.py your_song_path
```
