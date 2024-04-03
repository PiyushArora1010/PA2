import torch
import argparse
import librosa
import warnings

warnings.filterwarnings("ignore")

from module.models import modelDic
from module.utils import verification
from utils import load_checkpoint, EER_data, SISNR_SISDR, vox, punjabi, librimix

from speechbrain.pretrained import SepformerSeparation

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='wavlm_base_plus')
parser.add_argument('--checkpoint', type=str, default='checkpoints/wavlm_base_plus_nofinetune.pth')
parser.add_argument('--audio1', type=str, default='data/1.m4a')
parser.add_argument('--audio2', type=str, default='data/2.m4a')
parser.add_argument('--que', type=str, default="2.b")

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
    data = vox(16000, pathfile='data/wav/veri_test.txt')
    print("Size of the dataset: {}".format(len(data)))
    eer = EER_data(model, data)
    print("EER: {:.4f}".format(eer))

elif args.que == "1.d" or args.que == "1.e":
    model = modelDic[args.model]
    model = load_checkpoint(model, args.checkpoint)
    data = punjabi(4000, pathfile='data/kathbath/punjabi_meta/test_known_data.txt')
    print("Size of the dataset: {}".format(len(data)))
    eer = EER_data(model, data)
    print("EER: {:.4f}".format(eer))

elif args.que == "2.b":
    model = SepformerSeparation.from_hparams(
    "speechbrain/sepformer-libri2mix",
    run_opts={"device":"cuda"}
    ).to("cuda")

    data = librimix(1000, 'data/LibriMixData/test')

    print("Size of the dataset: {}".format(len(data)))

    sisnr, sisdr = SISNR_SISDR(model, data)

    print("SISNRi: {:.4f}".format(sisnr))
    print("SISDRi: {:.4f}".format(sisdr))
