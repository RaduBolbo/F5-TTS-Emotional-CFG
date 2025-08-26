# inference script.
from __future__ import annotations
import torch
import torchaudio
from ema_pytorch import EMA
from f5_tts.model import CFM  # Make sure CFM is correctly implemented for inference
from importlib.resources import files
import shutil
from datasets import load_dataset
from huggingface_hub import snapshot_download
import time
import pickle
from pathlib import Path
import json
import os
import random
from collections import defaultdict
from tqdm import tqdm
import gc
import os
import json
import torchaudio
from tqdm import tqdm

#from f5_tts.model import CFM, DiT, UNetT # old version without emotion conditioning
from f5_tts.model import CFMConditioned, DiTConditioned, UNetT, DiT # new version, with emotion conditioning
from f5_tts.model.utils import get_tokenizer
from ema_pytorch import EMA
from f5_tts.model.modules import MelSpec
import torch
from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

#-------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
emotion_dim = 128 # this should be set according to the mdoel configuration
emotion_conv_layers = 4 # this should be set according to the mdoel configuration
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'
faster_whisper_path = 'ckpts/resources/models/models--Systran--faster-whisper-large-v2-local'

tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)

emotion_dict = {
    "Angry": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
    "Happy": 5
}

# model params
wandb_resume_id = None
model_cls_emotion = DiTConditioned
#model_cfg_emotion = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, emotion_dim=128, conv_layers=emotion_conv_layers)
model_cls_pretrained = DiT
model_cfg_pretrained = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

def compute_mel_from_wav(
    audio_path: str,
    mel_spec_kwargs: dict,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute mel spectrogram from a .wav file using parameters in mel_spec_kwargs.
    """

    # Load and preprocess audio
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if sample_rate != mel_spec_kwargs["target_sample_rate"]:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=mel_spec_kwargs["target_sample_rate"]
        )
        audio = resampler(audio)

    audio = audio.to(device)

    # Initialize mel processor
    mel_processor = MelSpec(**mel_spec_kwargs).to(device)

    # Compute and return mel spectrogram
    mel = mel_processor(audio)  # (1, D, T)
    return mel.squeeze(0).permute(1, 0)  # (D, T)

class TTSModel:
    def __init__(self, model, vocoder_name, checkpoint_path: str, emotion_conditioning_parameters, device: str = "cuda"):
        self.device = device
        self.model = model  # Adjust if model args are needed
        #self.ema = EMA(self.model, include_online_model=False)
        self._load_checkpoint(checkpoint_path)
        self.emotion_conditioning_parameters = emotion_conditioning_parameters
        self.vocoder_name = vocoder_name

        self.vocoder = load_vocoder(vocoder_name=self.vocoder_name)



    def _load_checkpoint(self, path: str):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")

        if 'step' in checkpoint:
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        self.model = self.model.to(self.device)

        #self.ema_model.transformer.input_embed._initialize_proj_emotion_weights(self.emotion_conditioning_parameters['init_type'], self.emotion_conditioning_parameters['weight_reduction_scale']) 
        self.model.eval()
        del checkpoint
        #gc.collect()

    def remove_leading_value(self, spec, value=0.0):
        '''This removes the 'value' elements in the beginning of a melspectrogram'''
        gen_flat = spec[0]
        #reference_flat = reference[0]
        
        is_row_of_ones = torch.all(gen_flat == value, dim=1)
        num_rows_to_remove = torch.sum(is_row_of_ones).item()
        
        spec = spec[:, num_rows_to_remove:, :]
        
        return spec

    @torch.inference_mode()
    def infer(
        self,
        inference_text: str,
        inference_emotion: str,
        ref_mel: torch.Tensor,
        ref_text: str,
        ref_emotion: str,
        steps: int ,
        cfg_strength,
        cfg_strength2,
        sway_sampling_coef,
        seed: int = 50
    ) -> torch.Tensor:
        # Prepare conditioning
        text_input = [ref_text + ' ' + inference_text]
        emotion_input = [[ref_emotion, inference_emotion]]
        first_phrase_length = [len(ref_text)]  # or an integer if it's precomputed

        mel_lengths = torch.LongTensor([ref_mel.shape[0]])
        ref_audio_len = mel_lengths.item()
        estimated_duration = ref_audio_len + int(ref_audio_len * len(inference_text) / len(ref_text))

        start = time.perf_counter()
        if inference_emotion != None: # A) trained emotion model
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),  # (B, T, D)
                text=text_input,
                emotion=emotion_input,
                first_phrase_length=first_phrase_length,
                #duration=ref_duration,  # or adjust based on context
                duration = estimated_duration, # TOTAL DURARION ESTIMATED old_spec+new_part_spec = full_spec_length + vezi ca probabil nu-s bine initi ponderile + c vt
                steps=steps,
                cfg_strength=cfg_strength,
                cfg_strength2=cfg_strength2,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        else: # B) pretraiend emotionless mdoel
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),  # (B, T, D)
                text=text_input,
                #duration=ref_duration,  # or adjust based on context
                duration = estimated_duration, # TOTAL DURARION ESTIMATED old_spec+new_part_spec = full_spec_length + vezi ca probabil nu-s bine initi ponderile + c vt
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        
        end = time.perf_counter()

        generated_melspec = self.remove_leading_value(generated_melspec)

        # select only the part coresponding to the inference_text
        generated_melspec_2ndhalf = generated_melspec[:, ref_mel.shape[0]:, :]

        start = time.perf_counter()
        generated_audio = self.vocode(generated_melspec_2ndhalf)
        end = time.perf_counter()
        print(f'TIME vocoder ({len(text_input[0])}): ', end-start)

        return generated_melspec_2ndhalf, generated_audio

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(0) if mel.ndim == 2 else mel
        return self.vocoder.decode(mel.float().permute(0, 2, 1).to(self.device))

    @torch.inference_mode()
    def infer_custom(
        self,
        inference_text: str,
        inference_emotion: str,
        ref_mel: torch.Tensor,
        ref_text: str,
        ref_emotion: str,
        steps: int ,
        cfg_strength,
        cfg_strength2,
        sway_sampling_coef,
        seed: int = 50
    ) -> torch.Tensor:
        # Prepare conditioning
        text_input = [ref_text + ' ' + inference_text]
        emotion_input = [[ref_emotion, inference_emotion]]
        first_phrase_length = [len(ref_text)]  # or an integer if it's precomputed

        mel_lengths = torch.LongTensor([ref_mel.shape[0]])
        ref_audio_len = mel_lengths.item()
        estimated_duration = ref_audio_len + int(ref_audio_len * len(inference_text) / len(ref_text))

        if inference_emotion != None: # A) trained emotion model
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),  # (B, T, D)
                text=text_input,
                emotion=emotion_input,
                first_phrase_length=first_phrase_length,
                #duration=ref_duration,  # or adjust based on context
                duration = estimated_duration, # TOTAL DURARION ESTIMATED old_spec+new_part_spec = full_spec_length + vezi ca probabil nu-s bine initi ponderile + c vt
                steps=steps,
                cfg_strength=cfg_strength,
                cfg_strength2=cfg_strength2,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        else: # B) pretraiend emotionless mdoel
            generated_melspec, _ = self.model.sample(
                cond=ref_mel.to(self.device).unsqueeze(0),  # (B, T, D)
                text=text_input,
                #duration=ref_duration,  # or adjust based on context
                duration = estimated_duration, # TOTAL DURARION ESTIMATED old_spec+new_part_spec = full_spec_length + vezi ca probabil nu-s bine initi ponderile + c vt
                steps=steps,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
                seed=seed,
            )
        generated_melspec = self.remove_leading_value(generated_melspec)
        # select only the part coresponding to th inference_text
        generated_melspec_2ndhalf = generated_melspec[:, ref_mel.shape[0]:, :]
        generated_audio = self.vocode(generated_melspec_2ndhalf)

        return generated_melspec_2ndhalf, generated_audio

    def vocode(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.unsqueeze(0) if mel.ndim == 2 else mel
        return self.vocoder.decode(mel.float().permute(0, 2, 1).to(self.device))


if __name__ == '__main__':
    # ----------------- inference example ------------------
    ref_audio_path = "data/0011_angry.wav" # 
    ref_emotion = "Angry"
    ref_text = "The nine, the eggs, I keep."

    inference_text = "Hello, this is a text to check emotion."
    inference_emotion = "Surprise" # 'Angry', 'Surprise', 'Neutral', 'Sad', 'Happy'
    output_path = "data/output.wav"

    nfe = nfe_step # you can change it. Lower values lead to a faster inference, but a produce lower quality speech.
    cfg_strength2 = 10 # you can change it. Higher values may lead to stronger emotion representation, but also lower naturalness.
    
    # V1) text mirror + late fusion
    emotion_conditioning_parameters = {
        'emotion_condition_type': 'text_mirror', # 'text_mirror' = for the original variant in which emotion is passed throug a text-like structure, then concatenated along with text, cond and x and introduced to InputEmbedding that aggregates them all. This imples that weights of the Inputembedder are divided in old weights which are laoded from the checkpoint and the new weights that are Xavier initialized.
        'init_type': 'xavier_reduced',
        'weight_reduction_scale': 1,
        'emotion_dim': 128, 
        'emotion_conv_layers': 4,
        #'load_emotion_weights': True, # keep it True if not loadeing a modle that already has emotion conditioning. Make it False when loading a pretrained emotion model
        'load_emotion_weights': False, # keep it True if not loadeing a modle that already has emotion conditioning. Make it False when loading a pretrained emotion model
    }

    # V2) Cross Attention
    # emotion_conditioning_parameters = {
    #     'emotion_condition_type': 'cross_attention', # 'text_mirror' = for the original variant in which emotion is passed throug a text-like structure, then concatenated along with text, cond and x and introduced to InputEmbedding that aggregates them all. This imples that weights of the Inputembedder are divided in old weights which are laoded from the checkpoint and the new weights that are Xavier initialized.
    #     'emotion_dim': 1024, 
    #     'emotion_conv_layers': 4,
    #     'load_emotion_weights': True
    # }

    # V3) Text Early Fusion
    # emotion_conditioning_parameters = {
    #     'emotion_condition_type': 'text_early_fusion',
    #     'emotion_dim': 128,
    #     'emotion_conv_layers': 4,
    #     'load_emotion_weights': True,
    # }

    tokenizer_path = "ckpts/vocab.txt"
    vocab_char_map, vocab_size = get_tokenizer("EmiliaPetite_dataset_ZH_EN", "pinyin")
    device = 'cuda'

    checkpoint_path_emotion = 'ckpts/model_emo.pt'
    checkpoint_path_pretrained = 'ckpts/model_0.pt'

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model_cfg_emotion = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, emotion_dim=emotion_conditioning_parameters['emotion_dim'], conv_layers=emotion_conditioning_parameters['emotion_conv_layers'])

    # ----------------- instantiate model ------------------
    # A) Emotion model
    model_emotion = CFMConditioned(
        transformer=model_cls_emotion(**model_cfg_emotion, text_num_embeds=vocab_size, mel_dim=n_mel_channels, emotion_conditioning=emotion_conditioning_parameters),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    model_wraper_emotion = TTSModel(model_emotion, mel_spec_type, checkpoint_path_emotion, emotion_conditioning_parameters, device)

    # B) Pretrained emotionless model
    model_pretrained = CFM(
        transformer=model_cls_pretrained(**model_cfg_pretrained, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )
    model_wraper_pretrained = TTSModel(model_pretrained, mel_spec_type, checkpoint_path_pretrained, emotion_conditioning_parameters, device)
    
    mel = compute_mel_from_wav(ref_audio_path, mel_spec_kwargs, device="cuda")

    # A) emotion_model
    generated_melspec, generated_audio = model_wraper_emotion.infer(
        inference_text=inference_text,
        inference_emotion=inference_emotion,
        ref_mel=mel,
        ref_text=ref_text,
        ref_emotion=ref_emotion,
        steps=nfe,
        cfg_strength=cfg_strength,
        cfg_strength2=cfg_strength2,
        sway_sampling_coef=sway_sampling_coef
    )

    torchaudio.save(output_path.replace('.wav', f'_{inference_emotion}.wav'), generated_audio.cpu(), target_sample_rate)

    # B) pretrained emotionless model
    generated_melspec, generated_audio = model_wraper_pretrained.infer(
        inference_text=inference_text,
        inference_emotion=None, # not used for emotionlessmodel
        ref_mel=mel,
        ref_text=ref_text,
        ref_emotion=None, # not used for emotionlessmodel
        steps=nfe,
        cfg_strength=cfg_strength,
        cfg_strength2=None, # not used for emotionlessmodel
        sway_sampling_coef=sway_sampling_coef
    )

    torchaudio.save(output_path.replace('.wav', f'_NOemotion.wav'), generated_audio.cpu(), target_sample_rate)