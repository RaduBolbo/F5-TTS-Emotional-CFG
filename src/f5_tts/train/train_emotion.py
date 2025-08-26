# training script.

from importlib.resources import files

#from f5_tts.model import CFM, DiT, UNetT # old version without emotion conditioning
from f5_tts.model import CFMConditioned, DiTConditioned, UNetT # new version, with emotion conditioning
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer
from f5_tts.model.trainer_emotion import TrainerConditioned
from torch.utils.data import ConcatDataset

import copy


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

# train_dataset_path_esd = 'dataset/Emotional Speech Dataset (ESD)/train/dataset_descriptor.json' # ESD
# train_dataset_path_ravdess = 'dataset/RAVDESS/ravdess_metadata.json' # RAVDESS

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

batch_size_per_gpu = 384
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
    #'masking_type': 'original',
    'masking_type': '2nd_part_proportional_masked',

    'change_emotion_forward': False,

    'noise_2ndhalf': 'uniform', # other otpions than 'uniform' are proven o cause probelms at sample()

    # -----------------------
    # I) 'emotion_condition_type' 🎭
    # V0) Non-emotional
    # 'emotion_conditioning': {
    #     'emotion_condition_type': 'no_emotion_condition',
    # },

    # V1) text_mirror and late concatenation
    'emotion_conditioning': {
        'emotion_condition_type': 'text_mirror', 
        'init_type': 'xavier_reduced', 
        'weight_reduction_scale': 1, 
        #'emotion_dim': 16,
        'emotion_conv_layers': 4,
        #'load_emotion_weights': True, # keep it True if not loading a modle that already has emotion conditioning. Make it False when loading a pretrained emotion model
        'load_emotion_weights': False, # keep it True if not loading a modle that already has emotion conditioning. Make it False when loading a pretrained emotion model
    },

    # V2) cross attnetion
    # 'emotion_conditioning': {
    #     'emotion_condition_type': 'cross_attention',
    #     'emotion_dim': 1024,
    #     'emotion_conv_layers': 4,
    #     'load_emotion_weights': True
    # },

    # V3) text_early_fusion
    # 'emotion_conditioning': {
    #     'emotion_condition_type': 'text_early_fusion',
    #     'emotion_dim': 128,
    #     'emotion_conv_layers': 4,
    #     'load_emotion_weights': True,
    # },

    # -----------------------
    # II) Dataset parameters
    'emotion_conditioning_kwargs': {
        'emotions': {"Angry", "Neutral", "Sad", "Surprise", "Happy"}, # This is all the dataset: all emotions
        #'emotions': {"Sad", "Happy"}, # just 2 opposed emotoins
        #'emotions': {"Sad", "Happy", "Neutral"}, # just 2 opposed emotoins + Neutral
        'change_emotion_probability': 0.5, # If the emotion in the 1st and 2nd phrase differ .0 = never change emotion; 1 = always change emotion
        'same_sentence': False, # if True, the 1st and 2nd sentence will be the same phrase, from the same actor, but with distinct emotions
        'contrastive_loss': False, 
    }
}

emotion_conditioning_kwargs = copy.deepcopy(training_config['emotion_conditioning_kwargs'])
emotion_conditioning_val_kwargs = copy.deepcopy(training_config['emotion_conditioning_kwargs'])
emotion_conditioning_val_kwargs['contrastive_loss'] = False


# model params
if "F5TTS_Base" in exp_name:
    wandb_resume_id = None
    model_cls = DiTConditioned
    model_cfg = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, emotion_dim=training_config['emotion_conditioning']['emotion_dim'], conv_layers=training_config['emotion_conditioning']['emotion_conv_layers'])
elif exp_name == "E2TTS_Base":
    wandb_resume_id = None
    model_cls = UNetT
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)


# ----------------------------------------------------------------------- #


def main():
    if tokenizer == "custom":
        tokenizer_path = tokenizer_path
    else:
        tokenizer_path = train_dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

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
        #checkpoint_path=str(files("f5_tts").joinpath(f"../../ckpts/{exp_name}")),
        checkpoint_path=checkpoint_path,
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
