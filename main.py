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
parser.add_argument('--que', type=str, default="2.c")

args = parser.parse_args()

if args.que == "1.b":
    model = modelDic[args.model]
    model = load_checkpoint(model, args.checkpoint)
    data = vox(4000, pathfile='data/wav/veri_test.txt')
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
    source="speechbrain/sepformer-whamr", savedir='pretrained_models/sepformer-whamr',
    run_opts={"device":"cuda"}
    ).to("cuda")

    data = librimix(1000, 'data/LibriMixData/test')

    traindata, testdata = torch.utils.data.random_split(data, [int(len(data)*0.7), len(data)-int(len(data)*0.7)])

    print("Size of the Train dataset: {}".format(len(traindata)))
    print("Size of the Test dataset: {}".format(len(testdata)))

    sisnr, sisdr = SISNR_SISDR(model, testdata)

    print("SISNRi: {:.4f}".format(sisnr))
    print("SISDRi: {:.4f}".format(sisdr))

elif args.que == "2.c":
    model = SepformerSeparation.from_hparams(
    source="pretrained_models/finetuned",
    run_opts={"device":"cuda"}
    ).to("cuda")

    data = librimix(1000, 'data/LibriMixData/test')

    traindata, testdata = torch.utils.data.random_split(data, [int(len(data)*0.7), len(data)-int(len(data)*0.7)])

    print("Size of the Train dataset: {}".format(len(traindata)))
    print("Size of the Test dataset: {}".format(len(testdata)))

    sisnr, sisdr = SISNR_SISDR(model, testdata)

    print("SISNRi: {:.4f}".format(sisnr))
    print("SISDRi: {:.4f}".format(sisdr))