import os
import librosa
import dill
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

def wave2spec(file_list):
    for file in tqdm(file_list): 
        sr=16000 
        n_mels=128 
        n_fft=2048 
        hop_length=512 
        # DURATION = 30 
        
        artist_folder, artist, album, song, save_folder = file 
        artist_path = os.path.join(artist_folder, artist) 
        album_path = os.path.join(artist_path, album) 
        song_path = os.path.join(album_path, song) 

        # Create mel spectrogram and convert it to log scale 
        y, sr = librosa.load(song_path, sr=sr, mono=True) 
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length) 
        log_S = librosa.core.amplitude_to_db(S, ref=1.0) 
        data = (artist, log_S, song) 
        # Save each song 
        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song 
        with open(os.path.join(save_folder, save_name), 'wb') as fp: 
            dill.dump(data, fp)

def create_dataset_parellel(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, num_worker=10):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder+'/'+path)]

    all_list = []
    
    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in tqdm(artists):
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)
            for song in album_songs:
                song_path = os.path.join(album_path, song)
                all_list.append([artist_folder, artist, album, song, save_folder])
    
    length = len(all_list)/num_worker
    re_len = len(all_list) % num_worker


    list1 = [all_list.pop() for _ in range(re_len)]
    all_list = np.split(np.array(all_list), num_worker)
    all_list.append(np.array(list1))

    pool = mp.Pool(processes=num_worker+1)
    pool.map(wave2spec, all_list)
    pool.close()
    pool.join()

if __name__ == '__main__':    
    HERE = Path(__file__).resolve().parent      # .../<project>/CRNN
    ROOT = HERE.parent                          # .../<project>  ← CRNN 的上一層（artist20 在這層）

    # 1) 原曲
    art_dir  = ROOT / 'artist20' / 'train_val'                # <-- 指到 singer_x/album_y/ 的那層
    save_dir = HERE / 'song_data_artist20_origin'             # 存在 CRNN 裡（也可改 ROOT/...）
    save_dir.mkdir(parents=True, exist_ok=True)
    assert art_dir.is_dir(), f'找不到資料夾：{art_dir}'
    create_dataset_parellel(artist_folder=str(art_dir), save_folder=str(save_dir), num_worker=10)

    # 2) 人聲（若有 open-unmix 輸出）
    art_dir  = ROOT / 'artist20' / 'train_val'
    # art_dir  = ROOT / 'artist20_open_unmix_vocal' / 'train_val'
    save_dir = HERE / 'song_data_artist20_vocal'
    save_dir.mkdir(parents=True, exist_ok=True)
    assert art_dir.is_dir(), f'找不到資料夾：{art_dir}'
    create_dataset_parellel(artist_folder=str(art_dir), save_folder=str(save_dir), num_worker=10)

    # 3) 伴奏
    art_dir  = ROOT / 'artist20' / 'train_val'
    # art_dir  = ROOT / 'artist20_open_unmix_accomp' / 'train_val'
    save_dir = HERE / 'song_data_artist20_accomp'
    save_dir.mkdir(parents=True, exist_ok=True)
    assert art_dir.is_dir(), f'找不到資料夾：{art_dir}'
    create_dataset_parellel(artist_folder=str(art_dir), save_folder=str(save_dir), num_worker=10)