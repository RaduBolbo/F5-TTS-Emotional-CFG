# training script.

import argparse
import copy
import json

from f5_tts.model import CFMConditioned, DiTConditioned, UNetT
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.trainer_emotion import TrainerConditioned
from torch.utils.data import ConcatDataset


def _parse_args():
    p = argparse.ArgumentParser(
        description="Finetune F5-TTS with categorical conditioning. "
                    "Defaults match the in-file config; CLI flags override.",
        allow_abbrev=False,
    )
    p.add_argument("--train-descriptor", default=None,
                   help="Path to a single training descriptor JSON. "
                        "Overrides train_dataset_paths and forces dataset_keys=['ESD'].")
    p.add_argument("--val-descriptor", default=None,
                   help="Path to the validation descriptor JSON.")
    p.add_argument("--labels-file", default=None,
                   help="Path to a labels.json (from prepare_generic_dataset.py) "
                        "whose 'labels' list becomes the conditioning vocabulary.")
    p.add_argument("--labels", default=None,
                   help="Comma-separated label set. Overrides --labels-file.")
    p.add_argument("--change-label-prob", type=float, default=None,
                   help="Probability of sampling a second clip with a different label. "
                        "Set 0.0 for non-parallel datasets (default in-file: 0.5).")
    p.add_argument("--checkpoint-path", default=None,
                   help="Override the checkpoint directory.")
    return p.parse_args()


#-------------------------- Dataset Settings --------------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  # 'vocos' or 'bigvgan'
faster_whisper_path = 'ckpts/resources/models/models--Systran--faster-whisper-large-v2-local'

tokenizer = "pinyin"  # 'pinyin', 'char', or 'custom'
tokenizer_path = None  # if tokenizer = 'custom', define the path to the tokenizer you want to use (should be vocab.txt)
#dataset_name = "Emilia_ZH_EN"
train_dataset_name = "EmiliaPetite_dataset_ZH_EN"
val_dataset_name = 'EmiliaPetite_dataset_ZH_EN_val'

val_dataset_path = 'dataset/ESD/val/dataset_descriptor.json'

train_dataset_paths = {
    'ESD': 'dataset/ESD/train/dataset_descriptor.json',
    'RAVDESS': 'dataset/RAVDESS/ravdess_metadata.json', 
    'CREMA-D': 'dataset/CREMA-D/cremad_metadata.json', 
}

# -------------------------- Training Settings -------------------------- #

exp_name = "F5TTS_Base"  # F5TTS_Base | E2TTS_Base
wandb_name = "F5TTS_Emotion"


checkpoint_path = f"ckpts/{exp_name}"

learning_rate = 1e-5

batch_size_per_gpu = 2
batch_size_type = "sample"  # "frame" or "sample". Only "sample" is supported by the dataset_type="CustomDatasetConditioned"
max_samples = 64  # max sequences per batch if use frame-wise batch_size.
grad_accumulation_steps = 1  # note: updates = steps / grad_accumulation_steps
max_grad_norm = 1.0

epochs = 1000

save_per_updates = 50000  
last_per_steps = 10000  

training_config = {

    # 0) General-purpose parameters
    'freeze_backbone': False, # if true, it freezes 🧊 everything but the emotion embedding layer and the input embedding aggregator 
    'perform_validation': False, # if True, it performs validation for every epoch. Only use this if the model starts learning on the training datset
    'validation_numsteps': 1000, # Set to 'every_epoch' if you want val every epoch. how many steps until valdation occurs 
    'pre_valid': False, # set to true if validation is performed in the beginning of the training
    'compute_wer_valid': False,
    'compute_mcd_valid': False,
    'dataset_keys': ['ESD'],
    'masking_type': '2nd_part_proportional_masked',

    'change_emotion_forward': False,

    'noise_2ndhalf': 'uniform', # other otpions than 'uniform' are proven o cause probelms at sample()

    # -----------------------
    # I) 'emotion_condition_type':
    #   'no_emotion_condition' - baseline, no emotion signal
    #   'text_early_fusion'   - adds emotion embedding to text embedding (scaled by 0.1)
    #   'text_mirror'         - concatenates emotion to input projection
    #   'film'                - per-layer FiLM modulation (scale+shift) after each DiT block
    'emotion_conditioning': {
        'emotion_condition_type': 'film',
        'emotion_dim': 512,
        'emotion_conv_layers': 4,
        'load_emotion_weights': False, # True when adapting a pretrained non-emotion model; False when resuming an emotion-aware checkpoint
    },

    # -----------------------
    # II) Dataset parameters
    'emotion_conditioning_kwargs': {
        'emotions': {"Angry", "Neutral", "Sad", "Surprise", "Happy"},
        'change_emotion_probability': 0.5,
        'same_sentence': False,
        'contrastive_loss': False,
    }
}

# model params
emotion_cfg = training_config['emotion_conditioning']
if "F5TTS_Base" in exp_name:
    wandb_resume_id = None
    model_cls = DiTConditioned
    model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512,
        emotion_dim=emotion_cfg.get('emotion_dim', 100),
        conv_layers=emotion_cfg.get('emotion_conv_layers', 0),
    )
elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


# ----------------------------------------------------------------------- #


def main():
    cli = _parse_args()

    if cli.labels is not None:
        training_config['emotion_conditioning_kwargs']['emotions'] = {
            s.strip() for s in cli.labels.split(",") if s.strip()
        }
    elif cli.labels_file is not None:
        with open(cli.labels_file, "r", encoding="utf-8") as f:
            training_config['emotion_conditioning_kwargs']['emotions'] = set(json.load(f)["labels"])

    if cli.change_label_prob is not None:
        training_config['emotion_conditioning_kwargs']['change_emotion_probability'] = cli.change_label_prob

    if cli.train_descriptor is not None:
        train_dataset_paths.clear()
        train_dataset_paths['ESD'] = cli.train_descriptor
        training_config['dataset_keys'] = ['ESD']

    if cli.val_descriptor is not None:
        global val_dataset_path
        val_dataset_path = cli.val_descriptor

    ckpt_dir = cli.checkpoint_path or checkpoint_path

    emotion_conditioning_kwargs = copy.deepcopy(training_config['emotion_conditioning_kwargs'])
    emotion_conditioning_val_kwargs = copy.deepcopy(training_config['emotion_conditioning_kwargs'])
    emotion_conditioning_val_kwargs['contrastive_loss'] = False

    tok_path = tokenizer_path if tokenizer == "custom" else train_dataset_name
    vocab_char_map, vocab_size = get_tokenizer(tok_path, tokenizer)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFMConditioned(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels, emotion_conditioning=training_config['emotion_conditioning']),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    trainer = TrainerConditioned(
        model,
        epochs,
        learning_rate,
        num_warmup_updates=2,
        save_per_updates=save_per_updates,
        checkpoint_path=ckpt_dir,
        batch_size=batch_size_per_gpu,
        batch_size_type=batch_size_type,
        max_samples=max_samples,
        grad_accumulation_steps=grad_accumulation_steps,
        max_grad_norm=max_grad_norm,
        wandb_project="CFM-TTS",
        wandb_run_name=wandb_name,
        wandb_resume_id=wandb_resume_id,
        last_per_steps=last_per_steps,
        log_samples=True,
        mel_spec_type=mel_spec_type,
    )

    train_datasets = []
    for dataset_key in training_config['dataset_keys']:
        current_dataset = load_dataset(train_dataset_paths[dataset_key], tokenizer, dataset_type="CustomDatasetConditioned", mel_spec_kwargs=mel_spec_kwargs, emotion_conditioning_kwargs=emotion_conditioning_kwargs)
        print(f'{dataset_key} len = {len(current_dataset)}')
        train_datasets.append(current_dataset)
    train_dataset = ConcatDataset(train_datasets)

    val_dataset = load_dataset(val_dataset_path, tokenizer, dataset_type="CustomDatasetConditioned", mel_spec_kwargs=mel_spec_kwargs, emotion_conditioning_kwargs=emotion_conditioning_val_kwargs)

    trainer.train(
        train_dataset,
        val_dataset,
        resumable_with_seed=666,  # seed for shuffling dataset
        num_workers=1,
        faster_whisper_path=faster_whisper_path,
        **training_config
    )


if __name__ == "__main__":
    main()
