import torch
import torch.nn.functional as F
import librosa
from fairseq import tasks
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from torchaudio.transforms import Resample
from omegaconf import OmegaConf

def load_model(filepath):
    state = torch.load(filepath, map_location=lambda storage, loc: storage)
    # state = load_checkpoint_to_cpu(filepath)
    state["cfg"] = OmegaConf.create(state["cfg"])

    if "args" in state and state["args"] is not None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    elif "cfg" in state and state["cfg"] is not None:
        cfg = state["cfg"]
    else:
        raise RuntimeError(
            f"Neither args nor cfg exist in state keys = {state.keys()}"
            )

    task = tasks.setup_task(cfg.task)
    if "task_state" in state:
        task.load_state_dict(state["task_state"])

    model = task.build_model(cfg.model)

    return model, cfg, task

def verification(model, wav1, wav2, sr1=16000, sr2=16000, use_gpu=True):
    resample1 = Resample(orig_freq=sr1, new_freq=16000)
    resample2 = Resample(orig_freq=sr2, new_freq=16000)
    wav1 = resample1(wav1)
    wav2 = resample2(wav2)

    if use_gpu:
        model = model.cuda()
        wav1 = wav1.cuda()
        wav2 = wav2.cuda()

    model.eval()
    with torch.no_grad():
        emb1 = model(wav1)
        emb2 = model(wav2)

    sim = F.cosine_similarity(emb1, emb2)
    return sim