import os
import dill
import random
import itertools
from tqdm import tqdm
import multiprocessing as mp

import numpy as np
from numpy.random import RandomState
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import librosa
import librosa.display

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
from scipy import stats

from pathlib import Path
import json
from collections import defaultdict

def visualize_spectrogram(path, duration=None,
                          offset=0, sr=16000, n_mels=128, n_fft=2048,
                          hop_length=512):
    """This function creates a visualization of a spectrogram
    given the path to an audio file."""

    # Make a mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(path, sr=sr, duration=duration, offset=offset)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, n_fft=n_fft,
                                       hop_length=hop_length)

    # Convert to log scale (dB)
    log_S = librosa.logamplitude(S, ref_power=1.0)

    # Render output spectrogram in the console
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()


def create_dataset(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, only_vocal=False):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder+'/'+path)]

    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in artists:
        print(artist)
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)

            for song in album_songs:
                song_path = os.path.join(album_path, song)

                # Create mel spectrogram and convert it to the log scale
                y, sr = librosa.load(song_path, sr=sr)
                if only_vocal:
                    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
                    vocal_idx = get_vocal_idx(y, n_fft, hop_length)[:, 0]
                    S = S[:, vocal_idx]
                else:
                    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
                log_S = librosa.logamplitude(S, ref_power=1.0)
                data = (artist, log_S, song)

                # Save each song
                save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)


def create_dataset_mix_vocal(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, only_vocal=False):
    """This function creates the dataset given a folder
     with the correct structure (artist_folder/artists/albums/*.mp3)
    and saves it to a specified folder."""

    # get list of all artists
    os.makedirs(save_folder, exist_ok=True)
    artists = [path for path in os.listdir(artist_folder) if
               os.path.isdir(artist_folder+'/'+path)]

    # iterate through all artists, albums, songs and find mel spectrogram
    for artist in artists:
        print(artist)
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)

        for album in artist_albums:
            album_path = os.path.join(artist_path, album)
            album_songs = os.listdir(album_path)

            for song in album_songs:
                song_path = os.path.join(album_path, song)

                # Create mel spectrogram and convert it to the log scale
                y, sr = librosa.load(song_path, sr=sr)
                vocal_file = song_path.split('/')[-1].split('.')[0]+'.wav'
                vocal_pt = '/home/kevinco27/nas189/kevinco27/dataset/artist20_open_unmix_vocal/'+'/'.join(song_path.split('/')[7:-1])+'/'+vocal_file
                y_vocal, sr = librosa.load(vocal_pt, sr=sr)
                if only_vocal:
                    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
                    vocal_idx = get_vocal_idx(y_vocal, n_fft, hop_length)[:, 0]
                    vocal_idx = vocal_idx[np.where(vocal_idx < S.shape[1])]
                    S = S[:, vocal_idx]
                else:
                    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length)
                log_S = librosa.logamplitude(S, ref_power=1.0)
                data = (artist, log_S, song)

                # Save each song
                save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
                with open(os.path.join(save_folder, save_name), 'wb') as fp:
                    dill.dump(data, fp)


def create_dataset_parellel(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, only_vocal=False, num_worker=20):
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
    pool.join
    

def create_dataset_mix_vocal_parellel(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, only_vocal=False, num_worker=20):
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
    pool.map(wave2spec_mix_vocal, all_list)
    pool.close()
    pool.join
    

def create_dataset_non_vocal_parellel(artist_folder='artists', save_folder='song_data',
                   sr=16000, n_mels=128,
                   n_fft=2048, hop_length=512, only_vocal=False, num_worker=20):
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
    pool.map(wave2spec_non_vocal, all_list)
    pool.close()
    pool.join
    
    
def wave2spec(file_list):
    for file in tqdm(file_list):    
        sr=16000
        n_mels=128
        n_fft=2048
        hop_length=512
        only_vocal=True
        
        artist_folder, artist, album, song, save_folder = file
        artist_path = os.path.join(artist_folder, artist)
        album_path = os.path.join(artist_path, album)
        song_path = os.path.join(album_path, song)
        # Create mel spectrogram and convert it to the log scale
        y, sr = librosa.load(song_path, sr=sr)
        if only_vocal:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
            vocal_idx = get_vocal_idx(y, n_fft, hop_length)[:, 0]
            if len(vocal_idx) == 0:
                continue
            S = S[:, vocal_idx]
        else:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)
        data = (artist, log_S, song)

        # Save each song
        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
        with open(os.path.join(save_folder, save_name), 'wb') as fp:
            dill.dump(data, fp)


def wave2spec_mix_vocal(file_list):
    for file in tqdm(file_list):
        sr=16000
        n_mels=128
        n_fft=2048
        hop_length=512
        only_vocal=True
        
        artist_folder, artist, album, song, save_folder = file
        artist_path = os.path.join(artist_folder, artist)
        album_path = os.path.join(artist_path, album)
        song_path = os.path.join(album_path, song)
        # Create mel spectrogram and convert it to the log scale
        y, sr = librosa.load(song_path, sr=sr)
        vocal_file = song.split('.')[0]+'.wav'
        vocal_pt = f'/home/kevinco27/nas189/kevinco27/dataset/artist20_open_unmix_vocal/{artist}/{album}/{vocal_file}'
        y_vocal, sr = librosa.load(vocal_pt, sr=sr)
        if only_vocal:
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
            vocal_idx = get_vocal_idx(y_vocal, n_fft, hop_length)[:, 0]
            vocal_idx = vocal_idx[np.where(vocal_idx < S.shape[1])]
            S = S[:, vocal_idx]
        else:
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)
        data = (artist, log_S, song)

        # Save each song
        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
        with open(os.path.join(save_folder, save_name), 'wb') as fp:
            dill.dump(data, fp)
            

def wave2spec_non_vocal(file_list):
    for file in tqdm(file_list):
        sr=16000
        n_mels=128
        n_fft=2048
        hop_length=512
        only_vocal=True
        
        artist_folder, artist, album, song, save_folder = file
        artist_path = os.path.join(artist_folder, artist)
        album_path = os.path.join(artist_path, album)
        song_path = os.path.join(album_path, song)
        # Create mel spectrogram and convert it to the log scale
        y, sr = librosa.load(song_path, sr=sr)
        vocal_file = song.split('.')[0]+'.wav'
        vocal_pt = f'/home/kevinco27/nas189/kevinco27/dataset/artist20_open_unmix_vocal/{artist}/{album}/{vocal_file}'
        y_vocal, sr = librosa.load(vocal_pt, sr=sr)
        if only_vocal:
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
            vocal_idx = get_vocal_idx(y_vocal, n_fft, hop_length)[:, 0]
            vocal_idx = vocal_idx[np.where(vocal_idx < S.shape[1])]
            non_vocal_idx = list(set(range(S.shape[1])) - set(vocal_idx))
            import pdb;pdb.set_trace()
            S = S[:, non_vocal_idx]
        else:
            S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                            n_fft=n_fft,
                                            hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)
        data = (artist, log_S, song)

        # Save each song
        save_name = artist + '_%%-%%_' + album + '_%%-%%_' + song
        with open(os.path.join(save_folder, save_name), 'wb') as fp:
            dill.dump(data, fp)

                    
def get_vocal_idx(wave, n_fft, hop_length):
    stfts = np.abs(librosa.stft(wave, n_fft=n_fft, hop_length=hop_length)[None,:,:])
    db_stfts= librosa.core.amplitude_to_db(stfts)
    db_frames = np.sum(db_stfts, axis=1)
    db_avg = np.mean(db_frames, axis=1)
    db_std = np.std(db_frames, axis=1)
    return np.argwhere(db_frames[0]>db_avg+1*db_std)

                    
def load_dataset(song_folder_name='song_data',
                 artist_folder='artists',
                 voc_song_folder='voc_folder',
                 nb_classes=20, random_state=42):
    """This function loads the dataset based on a location;
     it returns a list of spectrograms
     and their corresponding artists/song names"""

    # Get all songs saved as numpy arrays in the given folder
    song_list = os.listdir(song_folder_name)
    if '.ipynb_checkpoints' in song_list:
        song_list.remove('.ipynb_checkpoints')
    # Load the list of artists
    artist_list = os.listdir(artist_folder)
    if '.ipynb_checkpoints' in artist_list:
        artist_list.remove('.ipynb_checkpoints')
    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    artist = []
    spectrogram = []
    song_name = []
    voc = []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        with open(os.path.join(song_folder_name, song), 'rb') as fp:
            loaded_song = dill.load(fp)
        with open(os.path.join(voc_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as vfp:
            loaded_voc = dill.load(vfp)
        if loaded_song[0] in artists:
            artist.append(loaded_song[0])
            spectrogram.append(loaded_song[1])
            song_name.append(loaded_song[2])
            voc.append(loaded_voc[1])

    return artist, spectrogram, song_name, voc

def load_dataset_album_split_dam(song_folder_name='song_data',
                             artist_folder='artists',
                             voc_song_folder='voc_data',
                             bgm_song_folder='bgm_data',
                             mel_song_folder='mel_data',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)
    # if '.ipynb_checkpoints' in song_list:
    #     song_list.remove('.ipynb_checkpoints')

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    train_albums = []
    test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        if '.ipynb_checkpoints' not in artist:
            albums = os.listdir(os.path.join(artist_folder, artist))
            random.shuffle(albums)
            test_albums.append(artist + '_%%-%%_' + albums.pop(0))
            val_albums.append(artist + '_%%-%%_' + albums.pop(0))
            train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []
    V_train, V_test, V_val = [], [], []
    B_train, B_test, B_val = [], [], []
    M_train, M_test, M_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        if '.ipynb_checkpoints' not in song:
            with open(os.path.join(song_folder_name, song), 'rb') as fp:
                loaded_song = dill.load(fp)

            with open(os.path.join(voc_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as vfp:
                voc_song = dill.load(vfp)

            with open(os.path.join(bgm_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as bfp:
                bgm_song = dill.load(bfp)

            mfp = os.path.join(mel_song_folder, song.split('.mp3')[0]+'.npy')
            mel_song = np.load(mfp)

            artist, album, song_name = song.split('_%%-%%_')
            artist_album = artist + '_%%-%%_' + album

            if loaded_song[0] in artists:
                if artist_album in train_albums:
                    Y_train.append(loaded_song[0])
                    X_train.append(loaded_song[1])
                    S_train.append(loaded_song[2])
                    V_train.append(voc_song[1]),
                    B_train.append(bgm_song[1])
                    M_train.append(mel_song)
                elif artist_album in test_albums:
                    Y_test.append(loaded_song[0])
                    X_test.append(loaded_song[1])
                    S_test.append(loaded_song[2])
                    V_test.append(voc_song[1])
                    B_test.append(bgm_song[1])
                    M_test.append(mel_song)

                elif artist_album in val_albums:
                    Y_val.append(loaded_song[0])
                    X_val.append(loaded_song[1])
                    S_val.append(loaded_song[2])
                    V_val.append(voc_song[1])
                    B_val.append(bgm_song[1])
                    M_val.append(mel_song)

    return Y_train, X_train, S_train, V_train, B_train, M_train,\
           Y_test, X_test, S_test, V_test, B_test, M_test,\
           Y_val, X_val, S_val, V_val, B_val, M_val

def load_dataset_album_split_da(song_folder_name='song_data',
                             artist_folder='artists',
                             voc_song_folder='voc_data',
                             bgm_song_folder='bgm_data',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)
    # if '.ipynb_checkpoints' in song_list:
    #     song_list.remove('.ipynb_checkpoints')

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    train_albums = []
    test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        if '.ipynb_checkpoints' not in artist:
            albums = os.listdir(os.path.join(artist_folder, artist))
            random.shuffle(albums)
            test_albums.append(artist + '_%%-%%_' + albums.pop(0))
            val_albums.append(artist + '_%%-%%_' + albums.pop(0))
            train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []
    V_train, V_test, V_val = [], [], []
    B_train, B_test, B_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        if '.ipynb_checkpoints' not in song:
            with open(os.path.join(song_folder_name, song), 'rb') as fp:
                loaded_song = dill.load(fp)

            with open(os.path.join(voc_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as vfp:
                voc_song = dill.load(vfp)

            with open(os.path.join(bgm_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as bfp:
                bgm_song = dill.load(bfp)

            artist, album, song_name = song.split('_%%-%%_')
            artist_album = artist + '_%%-%%_' + album

            if loaded_song[0] in artists:
                if artist_album in train_albums:
                    Y_train.append(loaded_song[0])
                    X_train.append(loaded_song[1])
                    S_train.append(loaded_song[2])
                    V_train.append(voc_song[1]),
                    B_train.append(bgm_song[1])
                elif artist_album in test_albums:
                    Y_test.append(loaded_song[0])
                    X_test.append(loaded_song[1])
                    S_test.append(loaded_song[2])
                    V_test.append(voc_song[1])
                    B_test.append(bgm_song[1])
                elif artist_album in val_albums:
                    Y_val.append(loaded_song[0])
                    X_val.append(loaded_song[1])
                    S_val.append(loaded_song[2])
                    V_val.append(voc_song[1])
                    B_val.append(bgm_song[1])

    return Y_train, X_train, S_train, V_train, B_train,\
           Y_test, X_test, S_test, V_test, B_test,\
           Y_val, X_val, S_val, V_val, B_val

def _json_paths_to_names(json_path: Path):
    """把 JSON 裡的 './train_val/artist/album/song.mp3' 轉成
       'artist_%%-%%_album_%%-%%_song.mp3' 的集合，用來和特徵檔名比對。"""
    with open(json_path, 'r') as f:
        rel_paths = json.load(f)  # 假設是 list[str]
    names = set()
    for p in rel_paths:
        p = p.replace('\\', '/').strip()
        parts = p.split('/')
        if len(parts) < 3:
            continue
        artist, album, song = parts[-3], parts[-2], parts[-1]
        names.add(f"{artist}_%%-%%_{album}_%%-%%_{song}")
    return names

# 防止OOM
def build_splits_from_json(
    json_folder: Path,
    song_folder: Path,            # 原始混音特徵（pickle，副檔名目前是 .mp3）
    voc_folder: Path = None,      # 人聲特徵（pickle，.wav）
    bgm_folder: Path = None,      # 伴奏特徵（pickle，.wav）
    mel_folder: Path = None       # Mel numpy（.npy）
):
    train_names = _json_paths_to_names(json_folder / 'train.json')
    val_names   = _json_paths_to_names(json_folder / 'val.json')

    # 做 label 對應（artist -> idx），用 train∪val 確保一致
    artists = sorted({n.split('_%%-%%_')[0] for n in (train_names + val_names)})
    label2idx = {a:i for i,a in enumerate(artists)}

    def name_to_paths(names):
        """回傳存在的各模態檔案的路徑與標籤；不載入內容"""
        X_orig, Y, S_song = [], [], []
        X_voc, X_bgm, X_mel = [], [], []
        miss_cnt = defaultdict(int)

        for nm in names:
            orig = song_folder / nm
            if not orig.exists():
                miss_cnt['orig'] += 1
                continue

            artist = nm.split('_%%-%%_')[0]
            label = label2idx[artist]
            song_stem = nm.split('.mp3')[0]  # 你現在的命名規則

            # 選配模態
            voc = (voc_folder / f"{song_stem}.wav") if voc_folder else None
            bgm = (bgm_folder / f"{song_stem}.wav") if bgm_folder else None
            mel = (mel_folder / f"{song_stem}.npy") if mel_folder else None

            X_orig.append(orig)
            Y.append(label)
            S_song.append(nm)  # 保留歌名（若未來要 song-level 投票用得到）

            if voc_folder and voc.exists(): X_voc.append(voc)
            if bgm_folder and bgm.exists(): X_bgm.append(bgm)
            if mel_folder and mel.exists(): X_mel.append(mel)

        return {
            'orig': X_orig,
            'voc' : X_voc,
            'bgm' : X_bgm,
            'mel' : X_mel,
            'y'   : Y,
            'song': S_song,
            'label2idx': label2idx
        }

    train_split = name_to_paths(train_names)
    val_split   = name_to_paths(val_names)

    return train_split, val_split, label2idx

# 會有OOM問題 不用
def load_dataset_from_json(
        json_folder,
        song_folder_name,          # 特徵檔資料夾（檔名如 artist_%%-%%_album_%%-%%_song.mp3）
        artist_folder,             # ROOT/artist20（放 train.json, val.json）
        voc_song_folder=None,      # 可選：人聲特徵檔資料夾（檔名同 stem + '.wav'）
        bgm_song_folder=None,      # 可選：伴奏特徵檔資料夾（檔名同 stem + '.wav'）
        mel_song_folder=None,      # 可選：mel .npy 特徵資料夾（檔名同 stem + '.npy'）
        nb_classes=20,
        random_state=42
    ):
    """以 train.json / val.json 決定分割（test 留空），讀入特徵檔。"""

    artist_folder = Path(artist_folder)
    song_folder_name = Path(song_folder_name)

    # 讀 JSON → 取得要放入 train / val 的「特徵檔名」集合
    train_names = _json_paths_to_names(json_folder / 'train.json')
    val_names   = _json_paths_to_names(json_folder / 'val.json')

    # 從 JSON 名單推得 artist 清單，再隨機挑 nb_classes 個（如需）
    all_names = train_names | val_names
    artist_list = sorted({ name.split('_%%-%%_')[0] for name in all_names })
    prng = RandomState(random_state)
    artists = set(artist_list)

    # 建回傳容器（test 一律空）
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []
    V_train, V_test, V_val = [], [], []
    B_train, B_test, B_val = [], [], []
    M_train, M_test, M_val = [], [], []

    # 便利特徵檔資料夾，只讀在 JSON 名單內的檔案
    for fname in os.listdir(song_folder_name):
        if fname.startswith('.') or fname == '.ipynb_checkpoints':
            continue
        # 確認在train還是val
        in_train = fname in train_names
        in_val   = fname in val_names
        if not (in_train or in_val):
            print("SKIP:" ,fname)
            continue  # 不在 split 名單→略過

        # 讀主特徵（pickle tuple： (artist, feature, song) ）
        with open(song_folder_name / fname, 'rb') as fp:
            loaded_song = dill.load(fp)
        artist_name = loaded_song[0]
        # 降維度
        feat = np.asarray(loaded_song[1], dtype=np.float16)
        if artist_name not in artists:
            continue

        # 其他對應特徵（若有提供路徑）：以相同基名組路徑
        stem = fname.split('.mp3')[0]  # 依你目前規則
        voc_song = bgm_song = None
        mel_song = None

        if voc_song_folder:
            vpath = Path(voc_song_folder) / (stem + '.wav')
            if vpath.exists():
                with open(vpath, 'rb') as vfp:
                    voc_song = dill.load(vfp)
        if bgm_song_folder:
            bpath = Path(bgm_song_folder) / (stem + '.wav')
            if bpath.exists():
                with open(bpath, 'rb') as bfp:
                    bgm_song = dill.load(bfp)
        if mel_song_folder:
            mpath = Path(mel_song_folder) / (stem + '.npy')
            if mpath.exists():
                mel_song = np.load(mpath)

        # 分流到 train / val（test 留空）
        if in_train:
            Y_train.append(loaded_song[0])
            X_train.append(feat) # loaded_song[1]
            S_train.append(loaded_song[2])
            if voc_song is not None: V_train.append(voc_song[1])
            if bgm_song is not None: B_train.append(bgm_song[1])
            if mel_song is not None: M_train.append(mel_song)

        elif in_val:
            Y_val.append(loaded_song[0])
            X_val.append(feat)
            S_val.append(loaded_song[2])
            if voc_song is not None: V_val.append(voc_song[1])
            if bgm_song is not None: B_val.append(bgm_song[1])
            if mel_song is not None: M_val.append(mel_song)

    print("train", len(Y_train), "val", len(Y_val))
    # 回傳介面與你原本一致（中段 test 皆為空）
    return (Y_train, X_train, S_train, V_train, B_train, M_train,
            Y_test,  X_test,  S_test,  V_test,  B_test,  M_test,
            Y_val,   X_val,   S_val,   V_val,   B_val,   M_val)


def load_dataset_album_split(song_folder_name='song_data',
                             artist_folder='artists',
                             voc_song_folder='voc_data',
                             nb_classes=20, random_state=42):
    """ This function loads a dataset and splits it on an album level"""
    song_list = os.listdir(song_folder_name)
    # if '.ipynb_checkpoints' in song_list:
    #     song_list.remove('.ipynb_checkpoints')

    # Load the list of artists
    artist_list = os.listdir(artist_folder)

    train_albums = []
    test_albums = []
    val_albums = []
    random.seed(random_state)
    for artist in os.listdir(artist_folder):
        if '.ipynb_checkpoints' not in artist:
            albums = os.listdir(os.path.join(artist_folder, artist))
            random.shuffle(albums)
            test_albums.append(artist + '_%%-%%_' + albums.pop(0))
            val_albums.append(artist + '_%%-%%_' + albums.pop(0))
            train_albums.extend([artist + '_%%-%%_' + album for album in albums])

    # select the appropriate number of classes
    prng = RandomState(random_state)
    artists = prng.choice(artist_list, size=nb_classes, replace=False)

    # Create empty lists
    Y_train, Y_test, Y_val = [], [], []
    X_train, X_test, X_val = [], [], []
    S_train, S_test, S_val = [], [], []
    V_train, V_test, V_val = [], [], []

    # Load each song into memory if the artist is included and return
    for song in song_list:
        if '.ipynb_checkpoints' not in song:
            with open(os.path.join(song_folder_name, song), 'rb') as fp:
                loaded_song = dill.load(fp)

            with open(os.path.join(voc_song_folder, song.split('.mp3')[0]+'.wav'), 'rb') as vfp:
                voc_song = dill.load(vfp)

            artist, album, song_name = song.split('_%%-%%_')
            artist_album = artist + '_%%-%%_' + album

            if loaded_song[0] in artists:
                if artist_album in train_albums:
                    Y_train.append(loaded_song[0])
                    X_train.append(loaded_song[1])
                    S_train.append(loaded_song[2])
                    V_train.append(voc_song[1])
                elif artist_album in test_albums:
                    Y_test.append(loaded_song[0])
                    X_test.append(loaded_song[1])
                    S_test.append(loaded_song[2])
                    V_test.append(voc_song[1])
                elif artist_album in val_albums:
                    Y_val.append(loaded_song[0])
                    X_val.append(loaded_song[1])
                    S_val.append(loaded_song[2])
                    V_val.append(voc_song[1])

    return Y_train, X_train, S_train, V_train,\
           Y_test, X_test, S_test, V_test,\
           Y_val, X_val, S_val, V_val


def load_dataset_song_split(song_folder_name='song_data',
                            artist_folder='artists',
                            voc_song_folder='voc_folder',
                            nb_classes=20,
                            test_split_size=0.1,
                            validation_split_size=0.1,
                            random_state=42):

    Y, X, S, V = load_dataset(song_folder_name=song_folder_name,
                           artist_folder=artist_folder,
                           voc_song_folder=voc_song_folder,
                           nb_classes=nb_classes,
                           random_state=random_state)
    # train and test split
    X_train, X_test, Y_train, Y_test, S_train, S_test, V_train, V_test = train_test_split(
        X, Y, S, V, test_size=test_split_size, stratify=Y,
        random_state=random_state)

    # Create a validation to be used to track progress
    X_train, X_val, Y_train, Y_val, S_train, S_val, V_train, V_val = train_test_split(
        X_train, Y_train, S_train, V_train, test_size=validation_split_size,
        shuffle=True, stratify=Y_train, random_state=random_state)

    return Y_train, X_train, S_train, V_train, \
           Y_test, X_test, S_test, V_test, \
           Y_val, X_val, S_val, V_val

def slice_songs_dam(X, Y, S, V, B, M, length=911):
   """Slices the spectrogram into sub-spectrograms according to length"""

   # Create empty lists for train and test sets
   artist = []
   spectrogram = []
   song_name = []
   voc = []
   bgm = []
   mel = []

   # Slice up songs using the length specified
   for i, song in enumerate(X):
       slices = int(song.shape[1] / length)
       for j in range(slices - 1):
           spectrogram.append(song[:, length * j:length * (j + 1)])
           artist.append(Y[i])
           song_name.append(S[i])
           voc.append(V[i][:, length * j:length * (j + 1)])
           bgm.append(B[i][:, length * j:length * (j + 1)])
           mel.append(M[i][:, length * j:length * (j + 1)])

   return np.array(spectrogram), np.array(artist), np.array(song_name), np.array(voc), np.array(bgm), np.array(mel)

def slice_songs_da(X, Y, S, V, B, length=911):
   """Slices the spectrogram into sub-spectrograms according to length"""

   # Create empty lists for train and test sets
   artist = []
   spectrogram = []
   song_name = []
   voc = []
   bgm = []

   # Slice up songs using the length specified
   for i, song in enumerate(X):
       #計算可切片數
       slices = int(song.shape[1] / length)
       for j in range(slices - 1):
           spectrogram.append(song[:, length * j:length * (j + 1)])
           artist.append(Y[i])
           song_name.append(S[i])
           #voc.append(V[i][:, length * j:length * (j + 1)])
           #bgm.append(B[i][:, length * j:length * (j + 1)])

   return spectrogram, artist, song_name, voc, bgm
   # return np.array(spectrogram), np.array(artist), np.array(song_name), np.array(voc), np.array(bgm)

def slice_songs(X, Y, S, V, length=911):
    """Slices the spectrogram into sub-spectrograms according to length"""

    # Create empty lists for train and test sets
    artist = []
    spectrogram = []
    song_name = []
    voc = []

    # Slice up songs using the length specified
    for i, song in enumerate(X):
        slices = int(song.shape[1] / length)
        for j in range(slices - 1):
            spectrogram.append(song[:, length * j:length * (j + 1)])
            artist.append(Y[i])
            song_name.append(S[i])
            voc.append(V[i][:, length * j:length * (j + 1)])
    
    return np.array(spectrogram), np.array(artist), np.array(song_name), np.array(voc)


def create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                             n_fft=2048, hop_length=512):
    """Create a spectrogram from a randomly selected song
     for each artist and plot"""

    # get list of all artists
    artists = os.listdir(artist_folder)

    fig, ax = plt.subplots(nrows=4, ncols=5, figsize=(14, 12), sharex=True,
                           sharey=True)

    row = 0
    col = 0

    # iterate through artists, randomly select an album,
    # randomly select a song, and plot a spectrogram on a grid
    for artist in artists:
        print(artist)
        # Randomly select album and song
        artist_path = os.path.join(artist_folder, artist)
        artist_albums = os.listdir(artist_path)
        album = random.choice(artist_albums)
        album_path = os.path.join(artist_path, album)
        album_songs = os.listdir(album_path)
        song = random.choice(album_songs)
        song_path = os.path.join(album_path, song)

        # Create mel spectrogram
        y, sr = librosa.load(song_path, sr=sr, offset=60, duration=3)
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels,
                                           n_fft=n_fft, hop_length=hop_length)
        log_S = librosa.logamplitude(S, ref_power=1.0)

        # Plot on grid
        plt.axes(ax[row, col])
        librosa.display.specshow(log_S, sr=sr)
        plt.title(artist)
        col += 1
        if col == 5:
            row += 1
            col = 0

    fig.tight_layout()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.get_cmap('Blues')):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_history(history, title="model accuracy"):
    """
    This function plots the training and validation accuracy
     per epoch of training
    """
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()

    return


def predict_artist(model, X, Y, S,
                   le, class_names,
                   slices=None, verbose=False,
                   ml_mode=False):
    """
    This function takes slices of songs and predicts their output.
    For each song, it votes on the most frequent artist.
    """
    print("Test results when pooling slices by song and voting:")
    # Obtain the list of songs
    songs = np.unique(S)

    prediction_list = []
    actual_list = []

    # Iterate through each song
    for song in songs:

        # Grab all slices related to a particular song
        X_song = X[S == song]
        Y_song = Y[S == song]

        # If not using full song, shuffle and take up to a number of slices
        if slices and slices <= X_song.shape[0]:
            X_song, Y_song = shuffle(X_song, Y_song)
            X_song = X_song[:slices]
            Y_song = Y_song[:slices]

        # Get probabilities of each class
        predictions = model.predict(X_song, verbose=0)

        if not ml_mode:
            # Get list of highest probability classes and their probability
            class_prediction = np.argmax(predictions, axis=1)
            class_probability = np.max(predictions, axis=1)

            # keep only predictions confident about;
            prediction_summary_trim = class_prediction[class_probability > 0.5]

            # deal with edge case where there is no confident class
            if len(prediction_summary_trim) == 0:
                prediction_summary_trim = class_prediction
        else:
            prediction_summary_trim = predictions

        # get most frequent class
        prediction = stats.mode(prediction_summary_trim)[0][0]
        actual = stats.mode(np.argmax(Y_song))[0][0]

        # Keeping track of overall song classification accuracy
        prediction_list.append(prediction)
        actual_list.append(actual)

        # Print out prediction
        if verbose:
            print(song)
            print("Predicted:", le.inverse_transform(prediction), "\nActual:",
                  le.inverse_transform(actual))
            print('\n')

    # Print overall song accuracy
    actual_array = np.array(actual_list)
    prediction_array = np.array(prediction_list)
    cm = confusion_matrix(actual_array, prediction_array)
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                          title='Confusion matrix for pooled results' +
                                ' with normalization')
    class_report = classification_report(actual_array, prediction_array,
                                         target_names=class_names)
    print(class_report)

    class_report_dict = classification_report(actual_array, prediction_array,
                                              target_names=class_names,
                                              output_dict=True)
    return (class_report, class_report_dict)


def encode_labels(Y, le=None, enc=None):
    """Encodes target variables into numbers and then one hot encodings"""

    # initialize encoders
    N = len(Y)

    # Encode the labels
    # A歌手→0、B歌手→1、....
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y).reshape(N, 1)
    else:
        Y_le = le.transform(Y).reshape(N, 1)

    # convert into one hot encoding
    if enc is None:
        enc = preprocessing.OneHotEncoder()
        Y_enc = enc.fit_transform(Y_le).toarray()
    else:
        Y_enc = enc.transform(Y_le).toarray()

    # return encoders to re-use on other data
    return Y_le, le, enc


def simple_encoding(Y, le=None):
    """Encodes target variables into numbers"""

    # initialize encoders
    N = Y.shape[0]

    # Encode the labels
    if le is None:
        le = preprocessing.LabelEncoder()
        Y_le = le.fit_transform(Y)
    else:
        Y_le = le.transform(Y)

    # return encoders to re-use on other data
    return Y_le, le


if __name__ == '__main__':

    # configuration options
    create_data = True
    create_visuals = False
    save_visuals = False

    if create_data:
        create_dataset(artist_folder='artists', save_folder='song_data',
                       sr=16000, n_mels=128, n_fft=2048,
                       hop_length=512)

    if create_visuals:
        # Create spectrogram for a specific song
        visualize_spectrogram(
            'artists/u2/The_Joshua_Tree/' +
            '02-I_Still_Haven_t_Found_What_I_m_Looking_For.mp3',
            offset=60, duration=29.12)

        # Create spectrogram subplots
        create_spectrogram_plots(artist_folder='artists', sr=16000, n_mels=128,
                                 n_fft=2048, hop_length=512)
        if save_visuals:
            plt.savefig(os.path.join('spectrograms.png'),
                        bbox_inches="tight")
