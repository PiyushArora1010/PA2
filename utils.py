import torch
import torchaudio
from module.models import ECAPA_TDNN_SMALL
from module.utils import verification
import numpy as np
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio
import librosa
import os

def load_checkpoint(model, checkpoint):
    model = ECAPA_TDNN_SMALL(**model)
    if checkpoint != 'none':
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def EER(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

class vox(torch.utils.data.Dataset):
    def __init__(self, cropLength, pathfile):
        self.cropLength = cropLength
        self.pathfile = pathfile
        
        self.audio1_list = []
        self.audio2_list = []
        self.labels = []

        file = open(pathfile, 'r')
        for line in file:
            line = line.strip().split()
            self.labels.append(int(line[0]))
            self.audio1_list.append(line[1])
            self.audio2_list.append(line[2])
            
    def __len__(self):
        return len(self.audio1_list)
    
    def __getitem__(self, idx):
        audio1, sr1 = torchaudio.load('data/wav/'+self.audio1_list[idx])
        audio2, sr2 = torchaudio.load('data/wav/'+self.audio2_list[idx])
        
        if audio1.size(1) < self.cropLength:
            audio1 = torch.nn.functional.pad(audio1, (0, self.cropLength - audio1.size(1)))
        if audio2.size(1) < self.cropLength:
            audio2 = torch.nn.functional.pad(audio2, (0, self.cropLength - audio2.size(1)))
        
        audio1 = audio1[:,:self.cropLength]
        audio2 = audio2[:,:self.cropLength]
        
        return audio1[0], audio2[0], self.labels[idx]

class punjabi(torch.utils.data.Dataset):
    def __init__(self, cropLength, pathfile, partition='test_known'):
        self.cropLength = cropLength
        self.pathfile = pathfile
        self.partition = partition
        self.DATA = 'data/kathbath/kb_data_clean_m4a/punjabi'
        self.audio1_list = []
        self.audio2_list = []
        self.labels = []

        file = open(pathfile, 'r')
        for line in file:
            line = line.strip().split()
            self.labels.append(int(line[0]))
            self.audio1_list.append(line[1])
            self.audio2_list.append(line[2])
        
        for i in range(len(self.audio1_list)):
            self.audio1_list[i] = self.audio1_list[i].split('punjabi')[1]
            self.audio1_list[i] = self.audio1_list[i].split('/')
            self.audio1_list[i] = self.partition + '/' + self.audio1_list[i][2] + '/' + self.audio1_list[i][4]
            self.audio1_list[i] = self.DATA + '/' + self.audio1_list[i]
            self.audio1_list[i] = self.audio1_list[i].replace('.wav', '.m4a')
        
        for i in range(len(self.audio2_list)):
            self.audio2_list[i] = self.audio2_list[i].split('punjabi')[1]
            self.audio2_list[i] = self.audio2_list[i].split('/')
            self.audio2_list[i] = self.partition + '/' + self.audio2_list[i][2] + '/' + self.audio2_list[i][4]
            self.audio2_list[i] = self.DATA + '/' + self.audio2_list[i]
            self.audio2_list[i] = self.audio2_list[i].replace('.wav', '.m4a')

    def __len__(self):
        return len(self.audio1_list)
    
    def __getitem__(self, idx):
        audio1, sr1 = librosa.load(self.audio1_list[idx])
        audio2, sr2 = librosa.load(self.audio2_list[idx])

        audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0)
        audio2 = torch.tensor(audio2, dtype=torch.float32).unsqueeze(0)
        
        if audio1.size(1) < self.cropLength:
            audio1 = torch.nn.functional.pad(audio1, (0, self.cropLength - audio1.size(1)))
        if audio2.size(1) < self.cropLength:
            audio2 = torch.nn.functional.pad(audio2, (0, self.cropLength - audio2.size(1)))
        
        audio1 = audio1[:,:self.cropLength]
        audio2 = audio2[:,:self.cropLength]
        
        return audio1[0], audio2[0], self.labels[idx]

class librimix(torch.utils.data.Dataset):
    def __init__(self, cropLen, datapath):
        self.cropLen = cropLen
        self.datapath = datapath
        self.sr1 = []
        self.sr2 = []
        self.mix = []

        mixfiles = datapath + '/mix_clean'
        s1files = datapath + '/s1'
        s2files = datapath + '/s2'

        for file in os.listdir(mixfiles):
            self.mix.append(mixfiles + '/' + file)
            self.sr1.append(s1files + '/' + file)
            self.sr2.append(s2files + '/' + file)

    def __len__(self):
        return len(self.mix)
    
    def __getitem__(self, idx):
        mix, sr = torchaudio.load(self.mix[idx])
        s1, sr = torchaudio.load(self.sr1[idx])
        s2, sr = torchaudio.load(self.sr2[idx])

        if mix.size(1) < self.cropLen:
            mix = torch.nn.functional.pad(mix, (0, self.cropLen - mix.size(1)))
        if s1.size(1) < self.cropLen:
            s1 = torch.nn.functional.pad(s1, (0, self.cropLen - s1.size(1)))
        if s2.size(1) < self.cropLen:
            s2 = torch.nn.functional.pad(s2, (0, self.cropLen - s2.size(1)))

        mix = mix[:,:self.cropLen]
        s1 = s1[:,:self.cropLen]
        s2 = s2[:,:self.cropLen]

        return mix[0], s1[0], s2[0]

def EER_data(model, data):

    model = model.eval()
    dataloader = torch.utils.data.DataLoader(data, batch_size=125, shuffle=False)
    count = 0
    eer = 1
    for x in tqdm(dataloader):
        labels = []
        preds = []
        audio1 = x[0].cuda()
        audio2 = x[1].cuda()
        label = x[2]

        sim = verification(model, audio1, audio2)
        labels.extend(label)
        sim = (sim.cpu().numpy() + 1)/2
        preds.extend(sim)
        eer_ = EER(preds, labels)
        count += 1
        eer = eer + (eer_ - eer)/count

    return eer

def SISNR_SISDR(model, data):
    model = model.eval()
    dataloader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)
    SISNR = ScaleInvariantSignalNoiseRatio().to('cuda')
    SISDR = ScaleInvariantSignalDistortionRatio().to('cuda')

    mean_SISNR_i = 0
    mean_SISDR_i = 0

    count = 0
    for x in tqdm(dataloader):
        mix = x[0].to('cuda')
        mix = mix - mix.mean()
        s1 = x[1].to('cuda')
        s2 = x[2].to('cuda')

        with torch.no_grad():
            extractedAudios = model.separate_batch(mix)
            audio1 = extractedAudios[:,:,0]
            audio2 = extractedAudios[:,:,1]

            SISNR_og = SISNR(audio1, mix) + SISNR(audio2, mix)
            SISDR_og = SISDR(audio1, mix) + SISDR(audio2, mix)

            SISNR_i = SISNR(audio1, s1) + SISNR(audio2, s2)
            SISDR_i = SISDR(audio1, s1) + SISDR(audio2, s2)

            SISNR_i = SISNR_og - SISNR_i
            SISDR_i = SISDR_og - SISDR_i

            SISNR_i = SISNR_i.mean().item()/2
            SISDR_i = SISDR_i.mean().item()/2

            count += 1

            mean_SISNR_i = mean_SISNR_i + (SISNR_i - mean_SISNR_i)/count
            mean_SISDR_i = mean_SISDR_i + (SISDR_i - mean_SISDR_i)/count

    return mean_SISNR_i, mean_SISDR_i