import argparse
import warnings

warnings.filterwarnings("ignore")

from module.models import modelDic
from module.utils import verification
from utils import load_checkpoint


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='wavlm_base_plus')
parser.add_argument('--checkpoint', type=str, default='checkpoints/wavlm_base_plus_nofinetune.pth')
parser.add_argument('--audio1', type=str, default='data/1.m4a')
parser.add_argument('--audio2', type=str, default='data/2.m4a')
parser.add_argument('--task', type=int, default=1)

args = parser.parse_args()

if args.task == 1:
    model = modelDic[args.model]
    model = load_checkpoint(model, args.checkpoint)
    verification(model, args.audio1, args.audio2)