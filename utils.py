import torch
import torchaudio
import numpy as np
import torchaudio.transforms as T
from module.models import ECAPA_TDNN_SMALL
from module.utils import verification
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

def load_checkpoint(model, checkpoint):
    model = ECAPA_TDNN_SMALL(**model)
    if checkpoint != 'none':
        state_dict = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict['model'], strict=False)
    return model

def EER(preds, labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, preds, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
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

def EER_data(model):

    data = vox(16000, pathfile='data/veri_test.txt')

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
        sim[sim>=0.5] = 1
        sim[sim<0.5] = 0
        preds.extend(sim)
        eer_ = EER(preds, labels)
        count += 1
        eer = eer + (eer_ - eer)/count

    return eer