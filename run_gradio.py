import gradio as gr
import numpy as np
import torch
from speechbrain.pretrained import SepformerSeparation
from module.models import modelDic
from module.utils import verification
from utils import load_checkpoint
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--que", type=str, default="2")
args = argparser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    if args.que == "2":
        model = SepformerSeparation.from_hparams(
            source="pretrained_models/finetuned",
            run_opts={"device":DEVICE}
        ).to(DEVICE) 

        def separate_audio(audio):
            global model
            sr = audio[0]
            audio = audio[1] / np.max(np.abs(audio[1]))
            audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            extractedAudios = model.separate_batch(audio)
            audio1 = extractedAudios[:,:,0].cpu().detach().numpy()
            audio2 = extractedAudios[:,:,1].cpu().detach().numpy()
            return (sr,audio1[0]), (sr,audio2[0])
        
        audio = gr.components.Audio(source="upload", label="Upload your audio file", type="numpy")
        outputs = [gr.outputs.Audio(label="Separated Audio 1", type="numpy"), gr.outputs.Audio(label="Separated Audio 2", type="numpy")]
        gr.Interface(fn=separate_audio, inputs=audio, outputs=outputs).launch()
    elif args.que == "1":
            model = modelDic['wavlm_base_plus']
            model = load_checkpoint(model, "checkpoints/wavlm_base_plus_finetune.pth")
            model = model.eval()

            def verify_audio(audio1, audio2):
                audio1 = audio1[1] / np.max(np.abs(audio1[1]))
                audio2 = audio2[1] / np.max(np.abs(audio2[1]))
                audio1 = torch.tensor(audio1, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                audio2 = torch.tensor(audio2, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                sim = verification(model, audio1, audio2)
                return str(sim.cpu().detach().numpy()[0])
            
            audio1 = gr.components.Audio(source="upload", label="Upload your first audio file", type="numpy")
            audio2 = gr.components.Audio(source="upload", label="Upload your second audio file", type="numpy")
            outputs = gr.outputs.Label(label="Similarity")
            gr.Interface(fn=verify_audio, inputs=[audio1, audio2], outputs=outputs).launch()
