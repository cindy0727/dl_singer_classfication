# dl singer identification

### Usage
Need to run extract_fea.py before running train_CRNN.py
#### extract_fea.py
Extracting melspectrograms of training data and validation data 
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
1. change the path of the folder in train_CRNN.py line 142, 143 and 144
```
142: json_folder = # path of artist20
143: artist_meta_dir = # path of training data and validation data
144: song_folder = # path of song_data_artist20_origin, same as extract_fea.py line 93
```
2. in terminal (need to run in Colab)
```
python train_CRNN.py --origin
```
#### testing.py
1. change the path of the folder in testing.py line 27 and 30
-Predicting test data and output json file
-I also uploaded my best model in folder "best model"
```
BEST_DIR = # path of best model
TEST_DIR = # path of testing data
```
```
python testing.py
```

### Reference
https://github.com/bill317996/Singer-identification-in-artist20.git
