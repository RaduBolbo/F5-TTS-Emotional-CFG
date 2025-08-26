from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
import torchaudio

from infer_emotion import (
    TTSModel,
    compute_mel_from_wav,
    CFMConditioned,
    DiTConditioned,
    CFM,
    DiT,
    get_tokenizer,
    n_fft,
    hop_length,
    win_length,
    n_mel_channels,
    target_sample_rate,
    mel_spec_type,         
    tokenizer,             
    cfg_strength,          
    nfe_step,              
    sway_sampling_coef,    
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLI emotion inference using F5-TTS-Emotional-CFG (emotion-conditioned)."
    )

    # --- Reference inputs ---
    p.add_argument("-ref", "--ref-audio-path", type=str, required=True,
                   help="Path to reference .wav for voice cloning.")
    p.add_argument("-rt", "--ref-text", type=str, required=True,
                   help="Transcription text for the reference audio.")
    p.add_argument("-re", "--ref-emotion", type=str, default="Neutral",
                   choices=["Angry", "Surprise", "Neutral", "Sad", "Happy"],
                   help='Reference emotion label (emotion in the reference audio). Can be one of: "Angry", "Surprise", "Neutral", "Sad", "Happy"')

    # --- Inference target ---
    p.add_argument("-it", "--inference-text", type=str, required=True,
                   help="New text to synthesize (will be appended after ref_text).")
    p.add_argument("-ie", "--inference-emotion", type=str, required=True,
                   choices=["Angry", "Surprise", "Neutral", "Sad", "Happy"],
                   help='Target emotion label for the new speech. "Angry", "Surprise", "Neutral", "Sad", "Happy"')

    # --- Output ---
    p.add_argument("-o", "--output-path", type=str, default="data/output.wav",
                   help="Path to the generated audio (.wav).")

    # --- Checkpoints ---
    p.add_argument("--checkpoint-path-emotion", type=str, default='ckpts/model_emo.pt',
                   help="Path to the trained emotion-conditioned model checkpoint (.pt).")

    # --- Tokenizer / vocab ---
    p.add_argument("--vocab-dataset-name", type=str, default="EmiliaPetite_dataset_ZH_EN",
                   help="Dataset name used for tokenizer building.")
    p.add_argument("--tokenizer", type=str, default=tokenizer,
                   choices=["pinyin", "char", "custom"],
                   help="Tokenizer type.")
    p.add_argument("--tokenizer-path", type=str, default=None,
                   help="Path to custom tokenizer vocab.txt (if tokenizer='custom').")

    # --- Sampling / guidance params ---
    p.add_argument("--nfe", type=int, default=nfe_step,
                   help="# function evaluations (steps). Lower = faster, lower quality.")
    p.add_argument("--cfg-strength", type=float, default=cfg_strength,
                   help="Classifier-free guidance for content/text.")
    p.add_argument("--cfg-strength2", type=float, default=10.0,
                   help="Emotion guidance strength; higher = stronger emotion, less natural.")
    p.add_argument("--sway-sampling-coef", type=float, default=sway_sampling_coef,
                   help="Sway sampling coefficient.")

    # --- Emotion conditioning block ---
    p.add_argument("--emotion-condition-type", type=str, default="text_mirror",
                   choices=["text_mirror", "cross_attention", "text_early_fusion"],
                   help="How emotion is injected into the transformer.")
    p.add_argument("--emotion-dim", type=int, default=128,
                   help="Dimension of emotion embedding.")
    p.add_argument("--emotion-conv-layers", type=int, default=4,
                   help="# of conv layers used in emotion path.")
    p.add_argument("--init-type", type=str, default="xavier_reduced",
                   help="(text_mirror only) initialization method for new emotion weights.")
    p.add_argument("--weight-reduction-scale", type=float, default=1.0,
                   help="(text_mirror only) scale for reduced Xavier init.")

    # --- Audio & mel spec ---
    p.add_argument("--mel-spec-type", type=str, default=mel_spec_type,
                   choices=["vocos", "bigvgan"], help="Vocoder/mel type.")
    p.add_argument("--target-sr", type=int, default=target_sample_rate)
    p.add_argument("--n-mel", type=int, default=n_mel_channels)
    p.add_argument("--n-fft", type=int, default=n_fft)
    p.add_argument("--hop-length", type=int, default=hop_length)
    p.add_argument("--win-length", type=int, default=win_length)

    # --- Device ---
    p.add_argument("--device", type=str, default='cuda',
                   choices=["cuda", "mps", "cpu"], help="Inference device.")

    return p


def main():
    args = build_arg_parser().parse_args()

    mel_spec_kwargs = dict(
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        n_mel_channels=args.n_mel,
        target_sample_rate=args.target_sr,
        mel_spec_type=args.mel_spec_type,
    )

    if args.tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("tokenizer='custom' requires --tokenizer-path (vocab.txt).")
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_dataset_name, args.tokenizer, args.tokenizer_path)
    else:
        vocab_char_map, vocab_size = get_tokenizer(args.vocab_dataset_name, args.tokenizer)

    emotion_conditioning_parameters = {
        "emotion_condition_type": args.emotion_condition_type,   # 'text_mirror' / 'cross_attention' / 'text_early_fusion'
        "init_type": args.init_type,
        "weight_reduction_scale": args.weight_reduction_scale,
        "emotion_dim": args.emotion_dim,
        "emotion_conv_layers": args.emotion_conv_layers,
        "load_emotion_weights": False,
    }

    model_cfg_emotion = dict(
        dim=1024, depth=22, heads=16, ff_mult=2,
        text_dim=512, emotion_dim=args.emotion_dim,
        conv_layers=args.emotion_conv_layers
    )

    transformer = DiTConditioned(
        **model_cfg_emotion,
        text_num_embeds=vocab_size,
        mel_dim=args.n_mel,
        emotion_conditioning=emotion_conditioning_parameters,
    )

    model_emotion = CFMConditioned(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    tts = TTSModel(
        model=model_emotion,
        vocoder_name=args.mel_spec_type,
        checkpoint_path=args.checkpoint_path_emotion,
        emotion_conditioning_parameters=emotion_conditioning_parameters,
        device=args.device,
    )

    mel = compute_mel_from_wav(args.ref_audio_path, mel_spec_kwargs, device=args.device)

    start = time.perf_counter()
    gen_mel, gen_audio = tts.infer(
        inference_text=args.inference_text,
        inference_emotion=args.inference_emotion,
        ref_mel=mel,
        ref_text=args.ref_text,
        ref_emotion=args.ref_emotion,
        steps=args.nfe,
        cfg_strength=args.cfg_strength,
        cfg_strength2=args.cfg_strength2,
        sway_sampling_coef=args.sway_sampling_coef,
    )
    dur = time.perf_counter() - start

    outpath = Path(args.output_path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(outpath), gen_audio.cpu(), args.target_sr)

    print(f"[OK] Saved: {outpath}  |  duration: {dur:.2f}s")
    print(f"    Inference emotion: {args.inference_emotion}")
    print(f"    Steps (nfe): {args.nfe} | cfg_strength: {args.cfg_strength} | cfg_strength2: {args.cfg_strength2}")


if __name__ == "__main__":
    main()
