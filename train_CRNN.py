import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import numpy as np
import os,time, json
import model
import h5py
import itertools
import utility
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import argparse
import pickle
import librosa
from pathlib import Path
# === Training 完成後做最終評估（使用最佳 F1 的模型） ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


class Dataset_4(Data.Dataset):
    def __init__(self, data_tensor, target_tensor1, target_tensor2, target_tensor3):
        assert data_tensor.size(0) == target_tensor1.size(0)
        self.data_tensor = data_tensor
        self.target_tensor1 = target_tensor1
        self.target_tensor2 = target_tensor2
        self.target_tensor3 = target_tensor3

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor1[index], self.target_tensor2[index], self.target_tensor3[index]

    def __len__(self):
        return self.data_tensor.size(0)

class Dataset_3(Data.Dataset):
    def __init__(self, data_tensor, target_tensor1, target_tensor2):
        assert data_tensor.size(0) == target_tensor1.size(0)
        self.data_tensor = data_tensor
        self.target_tensor1 = target_tensor1
        self.target_tensor2 = target_tensor2

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor1[index], self.target_tensor2[index]

    def __len__(self):
        return self.data_tensor.size(0)

# 用來把「輸入特徵 X」和「標籤 Y」包成能被 DataLoader 取用的物件
# 原本的
# class Dataset_2(Data.Dataset):
#     def __init__(self, data_tensor, target_tensor):
#         assert data_tensor.size(0) == target_tensor.size(0)
#         self.data_tensor = data_tensor
#         self.target_tensor = target_tensor

#     def __getitem__(self, index):
#         return self.data_tensor[index], self.target_tensor[index]

#     def __len__(self):
#         return self.data_tensor.size(0)
class Dataset_2(Data.Dataset):
    def __init__(self, xs, ys):
        assert len(xs) == len(ys)
        self.xs = xs              # list，每個元素是 2D numpy: (F, T)
        self.ys = ys              # list 或 1D array / int

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        x = self.xs[index]
        y = int(self.ys[index])
        # 轉成 float32 numpy，再轉 torch
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        elif x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

class Dataset_1(Data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


def main(classes_num=20, gid=0, random_state=0, \
            bs=100, learn_rate=0.0001, \
            val_num=1, stop_num=20,   #stop_num=200
            origin=True, vocal=False, remix=False,
            CRNN_model=True, CRNNx2_model=False,
            debug=False, TEST=False):

    start_time = time.time()

    save_folder = '../save/'+str(random_state)+'/'

    if origin and vocal and remix:
        save_folder = save_folder + '/all/'
    elif origin:
        save_folder = save_folder + '/ori/'
    elif vocal:
        save_folder = save_folder + '/voc/'
    elif remix:
        save_folder = save_folder + '/remix/'

    if not os.path.exists(save_folder+'/model/'):
        os.makedirs(save_folder+'/model/')
    if not os.path.exists(save_folder+'/result/'):
        os.makedirs(save_folder+'/result/')

    epoch_num = 50
    print('epoch_num: ', epoch_num)

    print('=======================')

    print('Loading CRNN model ...')


    if CRNN_model:
        Classifier = model.CRNN2D_elu(224,classes_num)
        Classifier.float()
        Classifier.cuda()
        Classifier.train()
    elif CRNNx2_model:
        Classifier = model.CRNN2D_elu2(288,classes_num)
        Classifier.float()
        Classifier.cuda()
        Classifier.train()

    print('Loading training data ...')

    CRNN_ROOT = Path(__file__).resolve().parent
    ROOT = CRNN_ROOT.parent
    json_folder = CRNN_ROOT / 'artist20'
    artist_meta_dir = CRNN_ROOT / 'artist20' / 'train_val'
    song_folder = CRNN_ROOT / 'song_data_artist20_origin'   # 原始混音
    
    
    # voc_folder  = CRNN_ROOT / 'song_data_artist20_vocal'    # 人聲
    # bgm_folder  = CRNN_ROOT / 'song_data_artist20_accomp'   # 伴奏(伴音)
    # json_folder = ROOT / 'artist20'
    # artist_meta_dir = ROOT / 'artist20' / 'train_val'

    # artist_folder=f'/home/bill317996/189/homes/kevinco27/dataset/artist20_mix'
    # song_folder=f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_mix'
    # voc_folder=f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_open_unmix_vocal_2'
    # bgm_folder = f'/home/bill317996/189/homes/kevinco27/ICASSP2020_meledy_extraction/music-artist-classification-crnn/song_data_open_unmix_kala'
    # random_states = [0,21,42]
    
    

    if debug:
        Y_train, X_train, S_train, V_train, B_train,\
        Y_test, X_test, S_test, V_test, B_test,\
        Y_val, X_val, S_val, V_val, B_val = \
        np.zeros(11437, dtype=int), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)), \
        np.zeros(11437, dtype=int), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)), \
        np.zeros(11437, dtype=int), np.zeros((11437, 128, 157)), np.zeros(11437), np.zeros((11437, 128, 157)), np.zeros((11437, 128, 157)) 

        Y_train[0] = 1
        Y_val[0] = 1
        Y_test[0] = 1
    else:
        # Y_train, X_train, S_train, V_train, B_train,\
        # Y_test, X_test, S_test, V_test, B_test,\
        # Y_val, X_val, S_val, V_val, B_val = \
        #     utility.load_dataset_album_split_da(song_folder_name=song_folder,
        #                                      artist_folder=artist_folder,
        #                                      voc_song_folder=voc_folder,
        #                                      bgm_song_folder=bgm_folder,
        #                                      nb_classes=classes_num,
        #                                      random_state=random_state)
        Y_train, X_train, S_train, V_train, B_train, M_train,\
        Y_test,  X_test,  S_test,  V_test,  B_test,  M_test,\
        Y_val,   X_val,   S_val,   V_val,   B_val,   M_val = \
            utility.load_dataset_from_json(
                                json_folder=json_folder,
                                song_folder_name=song_folder,
                                artist_folder=artist_meta_dir,
                                voc_song_folder=None,
                                bgm_song_folder=None,
                                mel_song_folder=None,     # 沒有就傳 None
                                nb_classes=20,
                                random_state=42)

    if not debug:
        print("Loaded and split dataset. Slicing songs...")

        slice_length = 157

        # Create slices out of the songs
        # X：mel spectrom
        # V_train：每段音訊的有效長度（給 RNN 做 pack/pad）
        # mask 或其他輔助特徵（如邊界、拍點資訊）
        X_test, Y_test, S_test, V_test, B_test = utility.slice_songs_da(X_test, Y_test, S_test, V_test, B_test,
                                                     length=slice_length)
        X_test = np.array(X_test, dtype=object)
        Y_test = np.array(Y_test, dtype=object)
        S_test = np.array(S_test, dtype=object)
        V_test = np.array(V_test, dtype=object)
        B_test = np.array(B_test, dtype=object)

        # print("Training set label counts:", np.unique(Y_train, return_counts=True))

        X_train, Y_train, S_train, V_train, B_train = utility.slice_songs_da(
                    X_train, Y_train, S_train, V_train, B_train, length=slice_length)
        X_val, Y_val, S_val, V_val, B_val = utility.slice_songs_da(
                    X_val, Y_val, S_val, V_val, B_val, length=slice_length)

    

        # # Encode the target vectors into one-hot encoded vectors
        # 幫歌手標號
        Y_train, le, enc = utility.encode_labels(Y_train)
        
        Y_val, le, enc = utility.encode_labels(Y_val, le, enc)

        Y_train = Y_train[:,0]
        Y_val = Y_val[:,0]

        if TEST:
            Y_test, le, enc = utility.encode_labels(Y_test, le, enc)
            Y_test = Y_test[:,0]
        else:
            n_classes = len(le.classes_)
            Y_test = np.zeros((0, n_classes), dtype=np.float32)

    
    if TEST:
        print(X_train.shape, Y_train.shape, S_train.shape, V_train.shape, B_train.shape)
        print(X_val.shape, Y_val.shape, S_val.shape, V_val.shape, B_val.shape)
        print(X_test.shape, Y_test.shape, S_test.shape, V_test.shape, B_test.shape)

    #####################################
    # numpy to tensor to data_loader
    # train
    # PyTorch 的模型吃的是 Tensor 
    # X_train = torch.from_numpy(X_train).float()
    # Y_train = torch.from_numpy(Y_train).long()
    # V_train = torch.from_numpy(V_train).float()
    # B_train = torch.from_numpy(B_train).float()

    # origin原始混音、vocal人聲-only、remix
    if origin:
        #　只用Ｘ和Ｙ
        original_set = Dataset_2(X_train, Y_train)
        original_loader = Data.DataLoader(dataset=original_set, batch_size=bs, shuffle=True)
    if vocal or remix:
        vocal_set = Dataset_2(data_tensor=V_train, target_tensor=Y_train)
        vocal_loader = Data.DataLoader(dataset=vocal_set, batch_size=bs, shuffle=True)
    if remix:
        bgm_set = Dataset_1(data_tensor=B_train)
        bgm_loader = Data.DataLoader(dataset=bgm_set, batch_size=bs, shuffle=True)


    # # val
    # if vocal and not origin:
    #     X_val = torch.from_numpy(V_val).float()
    #     Y_val = torch.from_numpy(Y_val).long()
    # else:
    #     X_val = torch.from_numpy(X_val).float()
    #     Y_val = torch.from_numpy(Y_val).long()

    val_set = Dataset_2(X_val, Y_val)
    val_loader = Data.DataLoader(dataset=val_set, batch_size=bs, shuffle=False)

    if TEST:
        # Test

        X_test = torch.from_numpy(X_test).float()
        Y_test = torch.from_numpy(Y_test).long()
        V_test = torch.from_numpy(V_test).float()

        test_o_set = Dataset_4(data_tensor=X_test, target_tensor1=Y_test, target_tensor2=S_test, target_tensor3=V_test)
        test_o_loader = Data.DataLoader(dataset=test_o_set, batch_size=bs, shuffle=False)

    #####################################

    best_epoch = 0
    best_F1 = 0
    best_ckpt_path = None   # 記錄最佳 checkpoint 路徑
    best_cloud_dir = None   # 記錄最佳雲端資料夾
    #Classifier = model.CRNN2D_elu(224,classes_num)

    CELoss = nn.CrossEntropyLoss()

    opt = optim.Adam(Classifier.parameters(),lr=learn_rate)

    print('Start training ...')

    start_time = time.time()
    early_stop_flag = False
    for epoch in range(epoch_num):
        if early_stop_flag:
            print('rs: ', random_state)
            print('Origin: ', origin, ' | Vocal: ', vocal, ' | Remix: ', remix)
            print('CRNN: ', CRNN_model, ' | CRNNx2: ', CRNNx2_model)
            print('     best_epoch: ', best_epoch, ' | best_val_F1: %.2f'% best_F1)
            if TEST:
                print('     Test original | frame level: %.2f'% test_F1_frame_o, ' | songs level: %.2f'% test_F1_songs_o)
            if vocal:
                print('     Test vocal | frame level: %.2f'% test_F1_frame_v, ' | songs level: %.2f'% test_F1_songs_v)
            break
        if stop_num:
            if epoch - best_epoch >= stop_num:
                early_stop_flag = True
                print('Early Stop!')
        all_loss = 0
        Classifier.train()

        if origin:
            # original_loader是X_train和Y_train
            for step, (batch_x, batch_y) in enumerate(original_loader):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
            
                logits, emb = Classifier(batch_x, batch_h)
                loss = CELoss(logits, batch_y)

                loss.backward()
                opt.step()

                all_loss += loss.item()
        if vocal:
            for step, (batch_x, batch_y) in enumerate(vocal_loader):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                

                pred_y, emb = Classifier(batch_x, batch_h)


                loss = CELoss(pred_y, batch_y)

                loss.backward()
                opt.step()

                all_loss += loss
        if remix:
            for step, ((batch_x, batch_y), batch_b) in enumerate(zip(vocal_loader,bgm_loader)):
                
                opt.zero_grad()

                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                batch_b = batch_b.cuda()

                batch_x = 10.0*torch.log10((10.0**(batch_x/10.0)) + (10.0**(batch_b/10.0)))
                
                pred_y, emb = Classifier(batch_x, batch_h)

                loss = CELoss(pred_y, batch_y)

                loss.backward()
                opt.step()

                all_loss += loss

        print('epoch: ', epoch, ' | Loss: %.4f'% all_loss, ' | time: %.2f'% (time.time()-start_time), '(s)')
        start_time = time.time()
        if epoch % val_num == 0:

            Classifier.eval()

            frame_true = []
            frame_pred = []

            with torch.no_grad():
                for step, (batch_x, batch_y) in enumerate(val_loader):
                    
                    batch_x = batch_x.cuda()
                    batch_y = batch_y.cuda()
                    batch_h = torch.randn(1, batch_x.size(0), 32).cuda()
                    
                    logits, emb = Classifier(batch_x, batch_h)          # ← logits
                    pred_idx = logits.argmax(dim=1).cpu().numpy()       # ← 直接對 logits 取 argmax
                    y_true   = batch_y.cpu().numpy()

                    frame_true.extend(y_true.tolist())
                    frame_pred.extend(pred_idx.tolist())

            val_F1 = f1_score(frame_true, frame_pred, average='weighted')
            print('     val F1: %.2f'% val_F1)
            
            # 找到最好的了  
            if best_F1 < val_F1:
                best_F1 = val_F1
                best_epoch = epoch

                print('     best_epoch: ', best_epoch, ' | best_val_F1: %.2f'% best_F1)

                torch.save({'Classifier_state_dict': Classifier.state_dict()
                            }, save_folder+'/model/CRNN2D_elu_model_state_dict_test')
                # 儲存到雲端
                RUN_NAME = time.strftime("CRNN_run_%Y%m%d_%H%M%S")
                CLOUD_DIR = Path('/content/drive/MyDrive/CRNN_runs') / RUN_NAME
                (CLOUD_DIR / 'model').mkdir(parents=True, exist_ok=True)
                (CLOUD_DIR / 'result').mkdir(parents=True, exist_ok=True)
                ckpt_path = CLOUD_DIR / 'model' / 'CRNN2D_elu_model_state_dict_test.pt'
                torch.save({
                    'epoch': int(best_epoch),
                    'best_val_F1': float(best_F1),
                    'model_name': 'CRNN2D_elu',
                    'classes_num': int(classes_num),
                    'flags': {'origin': bool(origin), 'vocal': bool(vocal), 'remix': bool(remix)},
                    'state_dict': Classifier.state_dict(),
                    # （可選）一起存 optimizer / scheduler 狀態，之後可無縫續訓
                    'optimizer': opt.state_dict() if 'opt' in globals() else None,
                }, str(ckpt_path))

                # 3) 存一份簡單的 meta.json（閱讀與追蹤更方便）
                meta = {
                    'best_epoch': int(best_epoch),
                    'best_val_F1': float(best_F1),
                    'slice_length': int(slice_length) if 'slice_length' in globals() else None,
                    'random_state': int(random_state) if 'random_state' in globals() else None,
                }
                with open(CLOUD_DIR / 'meta.json', 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                # （可選）若有 LabelEncoder，順手存起來，推論會用到
                try:
                    with open(CLOUD_DIR / 'label_encoder.pkl', 'wb') as f:
                        pickle.dump(le, f)
                except Exception as e:
                    print('[warn] 無法儲存 label encoder：', e)

                print('Saved to:', CLOUD_DIR)
                best_ckpt_path = ckpt_path
                best_cloud_dir = CLOUD_DIR


                frame_true = []
                frame_pred = []

                songs_true = []
                songs_pred = []

                songs_list = []

                songs_vote_dict = {}
                songs_true_dict = {}

                emb_list = []

                if TEST:
                    for step, (batch_x, batch_y, batch_song, batch_v) in enumerate(test_o_loader):
                        
                        batch_x = batch_x.cuda()
                        batch_y = batch_y.cuda()
                        batch_h = torch.randn(1, batch_x.size(0), 32).cuda()

                        pred_y, emb = Classifier(batch_x, batch_h)

                        pred_y = pred_y.detach().cpu().numpy()
                        batch_y = batch_y.detach().cpu().numpy()
                        emb = emb.detach().cpu().numpy()
                        batch_v = batch_v.detach().cpu().numpy()

                        for i in range(len(pred_y)):                
                            frame_true.append(batch_y[i])
                            frame_pred.append(np.argmax(pred_y[i]))

                            emb_list.append(emb[i])
                            
                            

                            onehot = np.zeros(20)
                            onehot[np.argmax(pred_y[i])] += 1

                            if batch_song[i] not in songs_list:
                                songs_list.append(batch_song[i])
                                songs_true_dict[batch_song[i]] = batch_y[i]
                                songs_vote_dict[batch_song[i]] = onehot

                            else:
                                songs_vote_dict[batch_song[i]] += onehot

                    for song in songs_list:
                        songs_true.append(songs_true_dict[song])
                        songs_pred.append(np.argmax(songs_vote_dict[song]))

                    np.savez(save_folder+'/result/ori_result.npz', \
                        pred=np.array(frame_pred), true=np.array(frame_true), emb=np.array(emb_list))

                        
                    test_F1_frame_o = f1_score(frame_true, frame_pred, average='weighted')
                    test_F1_songs_o = f1_score(songs_true, songs_pred, average='weighted')

                    print('     Test original | frame level: %.2f'% test_F1_frame_o, ' | songs level: %.2f'% test_F1_songs_o)

                    if vocal:
                        frame_true = []
                        frame_pred = []

                        songs_true = []
                        songs_pred = []

                        songs_list = []

                        songs_vote_dict = {}
                        songs_true_dict = {}

                        for step, (batch_x, batch_y, batch_song, batch_v) in enumerate(test_o_loader):
                            
                            batch_x = batch_v.cuda()
                            batch_y = batch_y.cuda()
                            batch_h = torch.randn(1, batch_x.size(0), 32).cuda()

                            pred_y, emb = Classifier(batch_x, batch_h)

                            pred_y = pred_y.detach().cpu().numpy()
                            batch_y = batch_y.detach().cpu().numpy()

                            for i in range(len(pred_y)):                
                                frame_true.append(batch_y[i])
                                frame_pred.append(np.argmax(pred_y[i]))

                                onehot = np.zeros(20)
                                onehot[np.argmax(pred_y[i])] += 1
                                
                                if batch_song[i] not in songs_list:
                                    songs_list.append(batch_song[i])
                                    songs_true_dict[batch_song[i]] = batch_y[i]
                                    songs_vote_dict[batch_song[i]] = onehot
                                else:
                                    songs_vote_dict[batch_song[i]] += onehot

                        for song in songs_list:
                            songs_true.append(songs_true_dict[song])
                            songs_pred.append(np.argmax(songs_vote_dict[song]))
                            
                        test_F1_frame_v = f1_score(frame_true, frame_pred, average='weighted')
                        test_F1_songs_v = f1_score(songs_true, songs_pred, average='weighted')
                        print('     Test vocal | frame level: %.2f'% test_F1_frame_v, ' | songs level: %.2f'% test_F1_songs_v)
                else:
                    print('[No test set] Skipping test evaluation (only using validation to pick the best epoch).')

    print("training end")
    # 1) 依出現順序建立歌曲 -> id 對照表
    uniq_order = {k: i for i, k in enumerate(dict.fromkeys(S_val))}
    # 2) 轉成與切片一一對應的 sid_val（長度 == len(X_val)）
    sid_val = np.array([uniq_order[k] for k in S_val], dtype=int)

    # 小檢查
    assert len(sid_val) == len(X_val)
    # 每首歌的切片數（可選，用來 debug）
    nslices_per_song = Counter(sid_val)
    print('#songs =', len(nslices_per_song), '; total slices =', len(X_val))

    # 確認同一首歌的標籤一致（非常重要）
    for sid in nslices_per_song.keys():
        ys = {int(Y_val[i]) for i in range(len(Y_val)) if sid_val[i] == sid}
        assert len(ys) == 1, f'song {sid} has multiple labels: {ys}'
    
    if best_ckpt_path is None:
        print("[warn] 尚未產生任何最佳模型（best_ckpt_path is None）。")
    else:
        print(f"\n[Final Eval] 以最佳模型評估：{best_ckpt_path}")

        # 載入最佳 checkpoint 並灌回模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        Classifier.load_state_dict(ckpt['state_dict'])
        Classifier.eval()
        song_logits_sum = defaultdict(lambda: 0)   # sid -> 累積 logits（tensor 或 numpy）
        song_counts     = defaultdict(int)         # sid -> 切片數
        song_true       = {}                       # sid -> 真實標籤(一次)


        # 逐 batch 推論，統計 top-1 / top-3 與混淆矩陣
        total = 0
        top1_correct = 0
        top3_correct = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            offset = 0  # 指向 sid_val 的讀取位置

            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                B = batch_y.size(0)

                # 取出對應這個 batch 的歌曲 id（依序對齊）
                sids = sid_val[offset: offset + B]
                offset += B

                h = torch.randn(1, B, 32, device=device)  # 依你的 GRU 隱層維度
                logits, _ = Classifier(batch_x, h)        # [B, C]

                # 累加到各歌曲
                for i in range(B):
                    sid = int(sids[i])
                    logit_i = logits[i].detach().cpu()
                    song_logits_sum[sid] = logit_i if song_logits_sum[sid] is None else song_logits_sum[sid] + logit_i
                    song_counts[sid]    += 1
                    song_true[sid]       = int(batch_y[i].item())

        # ===== 歌曲級 Top-1 / Top-3 =====
        all_true_song, all_pred_song = [], []
        top1_correct = top3_correct = 0

        for sid, logit_sum in song_logits_sum.items():
            mean_logits = logit_sum / song_counts[sid]
            pred1 = int(mean_logits.argmax().item())

            k = min(3, mean_logits.numel())
            topk_idx = torch.topk(mean_logits, k=k).indices.numpy().tolist()
            in_top3 = (song_true[sid] in topk_idx)

            all_true_song.append(song_true[sid])
            all_pred_song.append(pred1)
            top1_correct += int(pred1 == song_true[sid])
            top3_correct += int(in_top3)

        num_songs = len(song_true)
        top1_acc_song = top1_correct / num_songs
        top3_acc_song = top3_correct / num_songs
        print(f"[Song-level] #songs={num_songs}  Top-1={top1_acc_song:.4f}  Top-3={top3_acc_song:.4f}")

        # 1) 產生 cm（用訓練時的類別順序）
        labels = list(range(int(classes_num)))
        

        # 2) 類別名稱（若有 label encoder 就用名字）
        try:
            class_names = le.classes_.tolist()
        except Exception:
            class_names = [str(i) for i in labels]

        cm = confusion_matrix(all_true_song, all_pred_song, labels=labels)
        print("sum(cm_song) =", cm.sum())

        # 存成 CSV
        if best_cloud_dir is not None:
            cm_csv = best_cloud_dir / 'result' / 'confusion_matrix.csv'
            np.savetxt(str(cm_csv), cm, fmt='%d', delimiter=',')
            print("Confusion matrix CSV saved to:", cm_csv)

        # 存成圖（PNG）
        try:
            import matplotlib.pyplot as plt
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            fig, ax = plt.subplots(figsize=(10, 10))
            # 畫彩色熱圖
            disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)
            plt.title("Confusion Matrix (Song-level, Validation Set)")
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.xticks(rotation=90)   # x 軸標籤旋轉，避免歌手名字重疊
            cm_png = (best_cloud_dir / 'result' / 'confusion_matrix.png') if best_cloud_dir is not None else Path('confusion_matrix.png')
            plt.savefig(cm_png, dpi=150)
            plt.close()
            print("Confusion matrix PNG saved to:", cm_png)
        except Exception as e:
            print("[warn] 無法繪製/存檔混淆矩陣圖：", e)


def parser():
    
    p = argparse.ArgumentParser()

    p.add_argument('-class', '--classes_num', type=int, default=20)
    p.add_argument('-gid', '--gpu_index', type=int, default=0)
    p.add_argument('-bs', '--batch_size', type=int, default=100)
    p.add_argument('-lr', '--learn_rate', type=float, default=0.0001)
    p.add_argument('-val', '--val_num', type=int, default=1)
    p.add_argument('-stop', '--stop_num', type=int, default=20)

    p.add_argument('-rs', '--random_state', type=int, default=0)

    p.add_argument('--origin', dest='origin', action='store_true')
    p.add_argument('--vocal', dest='vocal', action='store_true')
    p.add_argument('--remix', dest='remix', action='store_true')
    p.add_argument('--all', dest='all', action='store_true')


    p.add_argument('--CRNNx2', dest='CRNNx2', action='store_true')

    p.add_argument('--debug', dest='debug', action='store_true')


    return p.parse_args()
if __name__ == '__main__':

    args = parser()

    classes_num = args.classes_num
    gid = args.gpu_index
    bs = args.batch_size
    learn_rate = args.learn_rate
    val_num = args.val_num
    stop_num = args.stop_num
    random_state = args.random_state

    origin = args.origin
    vocal = args.vocal
    remix = args.remix

    if args.all:
        origin = True
        vocal = True
        remix = True
    
    CRNNx2 = False
    if CRNNx2:
        CRNNx2_model = True
        CRNN_model = False
    else:
        CRNN_model = True
        CRNNx2_model = False
    debug = args.debug


    print('Singers classification with CRNN2D')
    print('Update in 20191016: artist20 ')

    print('=======================')
    print('classes_num', classes_num)
    print('gpu_index: ', gid, ' | random_state: ', random_state)
    print('bs: ',bs, ' | lr: %.5f'% learn_rate)
    print('val_num: ', val_num, ' | stop_num: ', stop_num)

    print('Origin: ', origin, ' | Vocal: ', vocal, ' | Remix: ', remix)
    
    print('CRNN: ', CRNN_model, ' | CRNNx2: ', CRNNx2_model)
    print('debug: ', debug)

    with torch.cuda.device(gid):
        main(classes_num=classes_num, gid=gid, random_state=random_state, \
            bs=bs, learn_rate=learn_rate, \
            val_num=val_num, stop_num=stop_num,
            origin=origin, vocal=vocal, remix=remix,
            CRNN_model=CRNN_model, CRNNx2_model=CRNNx2_model,
            debug=debug, TEST=False
            )

