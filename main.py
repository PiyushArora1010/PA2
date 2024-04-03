import torch

import argparse
import librosa
import warnings

warnings.filterwarnings("ignore")

from module.models import modelDic
from module.utils import verification
from utils import load_checkpoint, EER_data

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='wavlm_base_plus')
parser.add_argument('--checkpoint', type=str, default='checkpoints/wavlm_base_plus_nofinetune.pth')
parser.add_argument('--audio1', type=str, default='data/1.m4a')
parser.add_argument('--audio2', type=str, default='data/2.m4a')
parser.add_argument('--que', type=str, default="1.c")

args = parser.parse_args()

if args.que == "1.b":
    model = modelDic[args.model]
    model = load_checkpoint(model, args.checkpoint)
    audio1, sr1 = librosa.load(args.audio1)
    audio2, sr2 = librosa.load(args.audio2)
    audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0)
    audio2 = torch.tensor(audio2, dtype=torch.float32).unsqueeze(0)
    sim = verification(model, audio1, audio2, sr1, sr2)
    print("Similarity Score: {:.4f}".format(sim.item()))

elif args.que == "1.c":
    model = modelDic[args.model]
    model = load_checkpoint(model, args.checkpoint)
    eer = EER_data(model)
    print("EER: {:.4f}".format(eer))