from __future__ import annotations

import gc
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn, collate_fn_emotion
from f5_tts.model.utils import default, exists
from f5_tts.model.metrics import get_mcd_dtw, Calculate_MCD_from_ndarray, WhisperModelAdapter, get_error, replace_special_characters, link_words
import matplotlib.pyplot as plt
from jiwer import wer
import torch.nn.functional as F
import psutil

def get_memory_usage():
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_usage_mb = memory_info.rss / (1024 * 1024)
    
    return float(memory_usage_mb)


class TrainerConditioned:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        num_warmup_updates=20000,
        save_per_updates=1000,
        checkpoint_path=None,
        batch_size=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",  # "wandb" | "tensorboard" | None
        wandb_project="test_e2-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_steps=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",  # "vocos" | "bigvgan"
    ):

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and wandb.api.api_key:
            wandb.init(project=wandb_project, dir=f"ckpts/{wandb_run_name}/wandb")
        else:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name, "id": wandb_resume_id}}
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size": batch_size,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "gpus": self.accelerator.num_processes,
                    "noise_scheduler": noise_scheduler,
                },
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")
            print("summary writer at: ", f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.last_per_steps = default(last_per_steps, save_per_updates * grad_accumulation_steps)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_e2-tts")
        print('self.checkpoint_path :', self.checkpoint_path)

        self.batch_size = batch_size
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.vocoder_name = mel_spec_type

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=learning_rate)
        else:
            self.optimizer = AdamW(model.parameters(), lr=learning_rate)
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(self.optimizer).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                step=step,
            )
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            if last:
                print(f"Saved model as: ", f"{self.checkpoint_path}/model_last.pt")
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_last.pt")
                print(f"Saved last checkpoint at step {step}")
            else:
                print(f"{self.checkpoint_path}/model_{step}.pt")
                self.accelerator.save(checkpoint, f"{self.checkpoint_path}/model_{step}.pt")

    def load_checkpoint(self):
        if (
            not exists(self.checkpoint_path)
            or not os.path.exists(self.checkpoint_path)
            or not os.listdir(self.checkpoint_path)
        ):
            return 0

        self.accelerator.wait_for_everyone()
        if "model_last.pt" in os.listdir(self.checkpoint_path):
            latest_checkpoint = "model_last.pt"
        else:
            print('self.checkpoint_path: ', self.checkpoint_path)
            print(os.listdir(self.checkpoint_path))
            latest_checkpoint = sorted(
                [f for f in os.listdir(self.checkpoint_path) if f.endswith(".pt")],
                key=lambda x: int("".join(filter(str.isdigit, x))),
            )[-1]
        # checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", map_location=self.accelerator.device)  # rather use accelerator.load_state ಥ_ಥ
        checkpoint = torch.load(f"{self.checkpoint_path}/{latest_checkpoint}", weights_only=True, map_location="cpu")

        # patch for backward compatibility, 305e3ea
        for key in ["ema_model.mel_spec.mel_stft.mel_scale.fb", "ema_model.mel_spec.mel_stft.spectrogram.window"]:
            if key in checkpoint["ema_model_state_dict"]:
                del checkpoint["ema_model_state_dict"][key]

        if self.is_main:
            self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"], strict=False) # strict needs to be False in case of modifying the original model

        if "step" in checkpoint:
            # patch for backward compatibility, 305e3ea
            for key in ["mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
            self.accelerator.unwrap_model(self.optimizer).load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            step = checkpoint["step"]
        else:
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            step = 0

        del checkpoint
        gc.collect()
        return step

    def _save_current_model_sample(self, vocoder, ref_audio, ref_audio_len, text_inputs, first_phrase_length, emotion_inputs, mel_spec, mel_lengths, target_sample_rate, nfe_step, cfg_strength, sway_sampling_coef, ref_wav_path, gen_wav_path, masking_type):
        torchaudio.save(
            ref_wav_path, ref_audio.cpu(), target_sample_rate
        )
        cfg_strength2_list = [None, 10, 20]
        for emotion in [[[emotion_inputs[0][0], 'Happy']], [[emotion_inputs[0][0], 'Sad']], [[emotion_inputs[0][0], 'Angry']], [[emotion_inputs[0][0], 'Neutral']], [[emotion_inputs[0][0], 'Surprise']]]:
            for cfg_strength2 in cfg_strength2_list:
                gen_wav_path_new = gen_wav_path.replace(".wav", f"_{emotion[0][1]}_{cfg_strength2}.wav")
                if masking_type == 'original':
                    with torch.inference_mode():
                        
                        generated, _ = self.accelerator.unwrap_model(self.model).sample(
                            cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                            text=[" ".join([text_inputs[0], text_inputs[0]])],
                            first_phrase_length=first_phrase_length, 
                            emotion=emotion,
                            duration=ref_audio_len * 2,
                            steps=nfe_step,
                            cfg_strength=cfg_strength,
                            cfg_strength2=cfg_strength2,
                            sway_sampling_coef=sway_sampling_coef,
                            seed=50,
                        )
                        generated = generated.to(torch.float32)
                        gen_audio = vocoder.decode(
                            generated[:, ref_audio_len:, :].permute(0, 2, 1).to(self.accelerator.device)
                        )
                        print('emotion_inputs: ', emotion_inputs)
                elif masking_type == '2nd_part_proportional_masked':
                    cond = mel_spec[0, :int((first_phrase_length[0] * ref_audio_len / len(text_inputs[0])))] # select only the 1st part of the cond

                    generated, _ = self.accelerator.unwrap_model(self.model).sample(
                        cond=cond.unsqueeze(0),
                        text=[text_inputs[0]],
                        first_phrase_length=first_phrase_length, 
                        emotion=emotion,
                        duration=ref_audio_len,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        cfg_strength2=cfg_strength2,
                        sway_sampling_coef=sway_sampling_coef,
                        seed=50,
                    )
                    generated = generated.to(torch.float32)
                    gen_audio = vocoder.decode(
                        generated[:, :, :].permute(0, 2, 1).to(self.accelerator.device)
                    )


                torchaudio.save(
                    gen_wav_path_new, gen_audio.cpu(), target_sample_rate
                )

    def _return_dataloader(self, dataset_instance, batch_size, num_workers, collate_fn, generator, resumable_with_seed):
        if self.batch_size_type == "sample":
            dataloader_instance = DataLoader(
                dataset_instance,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_size=batch_size,
                shuffle=True,
                #shuffle=False,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False
            sampler = SequentialSampler(dataset_instance)
            batch_sampler = DynamicBatchSampler(
                sampler, batch_size, max_samples=self.max_samples, random_seed=resumable_with_seed, drop_last=False
            )
            dataloader_instance = DataLoader(
                dataset_instance,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}")
        return dataloader_instance

    def remove_leading_value(self, generated, reference, value=0.0):
        '''This removes the 'value' elements in the beginning of a melspectrogram'''
        gen_flat = generated[0]
        #reference_flat = reference[0]
        
        is_row_of_ones = torch.all(gen_flat == value, dim=1)
        num_rows_to_remove = torch.sum(is_row_of_ones).item()
        
        generated_trimmed = generated[:, num_rows_to_remove:, :]
        reference_trimmed = reference[:, num_rows_to_remove:, :]
        
        return generated_trimmed, reference_trimmed

    def _validation(self, global_step, dataloader, nfe_step, cfg_strength, sway_sampling_coef, vocoder, target_sample_rate, faster_whisper_path, asr_device, val_samples_path, compute_mcd=True, compute_wer=True, masking_type='original', noise_2ndhalf='uniform'):
        self.model.eval()
        mcd_scores = []
        wer_scores = []
        loss_scores = []

        text_language = 'en' # for the moment the model will only be trained for English

        if compute_mcd:
            mcd_toolbox = Calculate_MCD_from_ndarray(MCD_mode="dtw")
        if compute_wer:
            asr_tool = WhisperModelAdapter(faster_whisper_path, text_language, 'float16', 'cpu') # it doesn't fit on GPU for the moment -> to change device to 'cuda'

        print(f"Executing validation for {len(dataloader)} steps")
        val_step = 0
        for batch in tqdm(dataloader):
            val_step += 1
            gc.collect()
            torch.cuda.empty_cache()

            batch = batch[0]
            
            text_inputs = batch["text"]
            emotion_inputs = batch["emotion"]
            mel_spec = batch["mel"].permute(0, 2, 1).to('cuda')
            mel_lengths = batch["mel_lengths"].to('cuda')
            first_phrase_length = batch["first_phrase_length"]
            ref_audio_len = mel_lengths.item()

            loss, cond, pred, _, _ = self.model(
                mel_spec, text=text_inputs, first_phrase_length=first_phrase_length, emotion=emotion_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler, masking_type=masking_type, noise_2ndhalf=noise_2ndhalf
            )
            loss_scores.append(loss)

            if masking_type == 'original':
                generated, _ = self.accelerator.unwrap_model(self.model).sample(
                    cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                    text=[" ".join([text_inputs[0], text_inputs[0]])],
                    first_phrase_length=first_phrase_length, 
                    emotion=emotion_inputs,
                    duration=ref_audio_len * 2,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                gt_text = text_inputs[0]

                generated = generated.to(torch.float32)[:, ref_audio_len:, :] 
            elif masking_type == '2nd_part_proportional_masked':
                #cond = mel_spec[0, :first_phrase_length[0]]
                cond = mel_spec[0, :int((first_phrase_length[0] * ref_audio_len / len(text_inputs[0])))] # select only the 1st part of the cond

                generated, _ = self.accelerator.unwrap_model(self.model).sample(
                    cond=cond.unsqueeze(0),
                    text=[text_inputs[0]], 
                    first_phrase_length=first_phrase_length, 
                    emotion=emotion_inputs,
                    duration=ref_audio_len,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )
                gt_text = text_inputs[0][first_phrase_length[0]:] # In this case only, the 2nd half of the phrase is considered

                generated = generated.to(torch.float32)[:, int((first_phrase_length[0] * ref_audio_len / len(text_inputs[0]))):, :] # only select the generated piece of melspec.

            # extract the surplus zeros in the melspectrogram that are introducing by reference masking
            generated, mel_spec = self.remove_leading_value(generated, mel_spec)

            gen_audio = vocoder.decode(
                generated.permute(0, 2, 1).to(self.accelerator.device)
            )
            ref_audio = vocoder.decode(
                mel_spec.permute(0, 2, 1).to(self.accelerator.device)
            )

            if val_step % 1 == 0:
                gen_wav_path = os.path.join(val_samples_path, f'{val_step}_gen.wav')
                ref_wav_path = os.path.join(val_samples_path, f'{val_step}_ref.wav')
                self._save_current_model_sample(vocoder, ref_audio, ref_audio_len, text_inputs, first_phrase_length, emotion_inputs, mel_spec, mel_lengths, target_sample_rate, nfe_step, cfg_strength, sway_sampling_coef, ref_wav_path, gen_wav_path, masking_type)

            if compute_wer:
                transcription_text = asr_tool.execute_asr_faster(
                    wav_or_filepath=gen_audio.squeeze(0).to('cpu').numpy(),
                    language=text_language,
                    audios_sample_rate=target_sample_rate,
                )
                if text_language == "en":
                    error_phrase = replace_special_characters(gt_text, lang=text_language).lower()
                    transcription_text = replace_special_characters(transcription_text, lang=text_language).lower()
                    error_phrase = link_words(error_phrase, transcription_text)
                    transcription_text = link_words(transcription_text, error_phrase)

                    error = get_error(error_phrase, transcription_text, wer)
                    wer_scores.append(error)
                else:
                    raise ValueError("Only En is supported for WER")

            util_ref_audio = ref_audio[:, -gen_audio.shape[1]:] # select only that piece of reference that corresponds to the generated audio
            if compute_mcd:
                mcd = mcd_toolbox.calculate_mcd(util_ref_audio.squeeze(0).to('cpu').numpy(), gen_audio.squeeze(0).to('cpu').numpy(), audios_sample_rate=target_sample_rate)
                mcd_scores.append(mcd)

            verbose = False
            if verbose: # if verbose, save generated melspecs and audios to inspect manually
                mel_spec_image = mel_spec[0].to('cpu').numpy()
                mel_spec_image = ((mel_spec_image - mel_spec_image.min()) / (mel_spec_image.max() - mel_spec_image.min()) * 255).astype('uint8')

                generated_image = generated[0].to('cpu').numpy()
                generated_image = ((generated_image - generated_image.min()) / (generated_image.max() - generated_image.min()) * 255).astype('uint8')

                # Plot the spectrograms
                plt.imshow(mel_spec_image, cmap="hot")
                plt.title("Reference melspec")
                plt.colorbar()
                plt.savefig("REF.png")
                plt.close()

                plt.imshow(generated_image, cmap="hot")
                plt.title("Generated melspec")
                plt.colorbar()
                plt.savefig("GEN.png")
                plt.close()

                # Test audio
                torchaudio.save(
                    'GEN_wav.wav', ref_audio.cpu(), target_sample_rate
                )
                torchaudio.save(
                    'REF_wav.wav', gen_audio.cpu(), target_sample_rate
                )

        if compute_mcd:
            average_mcd = sum(mcd_scores) / len(dataloader)
            print('val average_mcd: ', average_mcd)
            self.accelerator.log({"MCD": average_mcd}, step=global_step)
        if compute_wer:
            average_wer = sum(wer_scores) / len(dataloader)
            print('val average_wer: ', average_wer)
            self.accelerator.log({"WER": average_wer}, step=global_step)
                        
        average_loss = sum(loss_scores) / len(dataloader) 
        print('val average_loss: ', average_loss)
        self.accelerator.log({"val loss": average_loss}, step=global_step)

        self.model.train()

    def train(self, train_dataset: Dataset, val_dataset: Dataset, faster_whisper_path, num_workers=16, resumable_with_seed: int = None, masking_type='original', **kwargs):
        print("01 self.optimizer.param_groups[0]['lr']: ", self.optimizer.param_groups[0]['lr'])

        
        if self.log_samples:
            from f5_tts.infer.utils_infer import cfg_strength, load_vocoder, nfe_step, sway_sampling_coef

            vocoder = load_vocoder(vocoder_name=self.vocoder_name)
            target_sample_rate = self.accelerator.unwrap_model(self.model).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples_train_sample"
            val_samples_path = f"{self.checkpoint_path}/samples_val"
            # forward_samples_path = f"{self.checkpoint_path}/samples_train_forward"
            os.makedirs(log_samples_path, exist_ok=True)
            os.makedirs(val_samples_path, exist_ok=True)
            # os.makedirs(forward_samples_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        train_dataloader = self._return_dataloader(train_dataset, self.batch_size, num_workers, collate_fn_emotion, generator,resumable_with_seed)
        val_dataloader = self._return_dataloader(val_dataset, 1, num_workers, collate_fn_emotion, generator, resumable_with_seed)

        #  accelerator.prepare() dispatches batches to devices;
        #  which means the length of dataloader calculated before, should consider the number of devices
        warmup_steps = (
            self.num_warmup_updates * self.accelerator.num_processes
        )  # consider a fixed warmup steps while using accelerate multi-gpu ddp
        # otherwise by default with split_batches=False, warmup steps change with num_processes
        total_steps = len(train_dataloader) * self.epochs / self.grad_accumulation_steps
        decay_steps = total_steps - warmup_steps
        warmup_scheduler = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps)
        decay_scheduler = LinearLR(self.optimizer, start_factor=1.0, end_factor=1e-6, total_iters=decay_steps)
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
        )
        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )  # actual steps = 1 gpu steps / gpus
        start_step = self.load_checkpoint()
        global_step = start_step

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(train_dataloader, num_batches=skipped_batch)
        else:
            skipped_epoch = 0

        ####
        # Modifying the weights of the InputEmbedding's instance (see dit_emotion.py)
        ####
        if kwargs['emotion_conditioning']['emotion_condition_type'] == 'text_mirror':
            init_type = kwargs['emotion_conditioning']['init_type']
            weight_reduction_scale = kwargs['emotion_conditioning']['weight_reduction_scale']
            if kwargs['emotion_conditioning']['load_emotion_weights']:
                self.model.transformer.input_embed._initialize_proj_emotion_weights(init_type, weight_reduction_scale)

        freeze_backbone = kwargs['freeze_backbone']
        if freeze_backbone:
            #if kwargs['emotion_conditioning']['emotion_condition_type'] == 'text_mirror':
                print('Freezing network 🧊 ...')
                # Freeze all parameters in self.model.transformer
                for param in self.model.transformer.parameters():
                    param.requires_grad = False

                # Unfreeze the parameters of input_embed and emotion_embed
                for param in self.model.transformer.input_embed.parameters():
                    param.requires_grad = True

                for param in self.model.transformer.emotion_embed.parameters():
                    param.requires_grad = True

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar = tqdm(
                    skipped_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                    initial=skipped_batch,
                    total=orig_epoch_step,
                )
            else:
                progress_bar = tqdm(
                    train_dataloader,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    unit="step",
                    disable=not self.accelerator.is_local_main_process,
                )


            asr_device = 'cpu'
            if kwargs['perform_validation'] and kwargs['pre_valid']:
                with torch.no_grad():
                    self._validation(global_step, val_dataloader, nfe_step, cfg_strength, sway_sampling_coef, vocoder, target_sample_rate, faster_whisper_path, asr_device, val_samples_path, kwargs['compute_mcd_valid'], kwargs['compute_wer_valid'], masking_type, kwargs['noise_2ndhalf'])


            for batch in progress_bar:
                '''
                Each batch will be a dict where each filed is a list of lists: the outer list contains the elements in the batch and the
                inner list have one element iof contrastive_loss is false and 2 elements if contrastive_loss 
                is True (here the inner list elements share the same phrases, but differ in the 2nd emotion)
                '''
                
                with self.accelerator.accumulate(self.model):

                    
                    if kwargs['emotion_conditioning_kwargs']['contrastive_loss']:
                        batch2 = batch[1]
                        batch = batch[0]             
                    else:
                        batch = batch[0]           

                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    first_phrase_length = batch["first_phrase_length"]
                    emotion = batch["emotion"]
                    if kwargs['emotion_conditioning_kwargs']['contrastive_loss']:
                        text_inputs2 = batch2["text"]
                        mel_spec2 = batch2["mel"].permute(0, 2, 1)
                        mel_lengths2 = batch2["mel_lengths"]
                        first_phrase_length2 = batch2["first_phrase_length"]
                        emotion2 = batch2["emotion"]

                    # TODO. add duration predictor training
                    if self.duration_predictor is not None and self.accelerator.is_local_main_process:
                        dur_loss = self.duration_predictor(mel_spec, lens=batch.get("durations"))
                        self.accelerator.log({"duration loss": dur_loss.item()}, step=global_step)
                    
                    if kwargs['emotion_conditioning_kwargs']['contrastive_loss'] or kwargs['change_emotion_forward']:
                        # **** Should have done this inside the collate function
                        def pad_to_match(tensor1, tensor2):
                            x1, x2 = tensor1.shape[1], tensor2.shape[1]
                            
                            if x1 < x2:
                                pad_size = x2 - x1
                                padded_tensor1 = F.pad(tensor1, (0, 0, 0, pad_size))  # Pad only the X dimension
                                return padded_tensor1, tensor2
                            elif x2 < x1:
                                pad_size = x1 - x2
                                padded_tensor2 = F.pad(tensor2, (0, 0, 0, pad_size))  # Pad only the X dimension
                                return tensor1, padded_tensor2
                            else:
                                return tensor1, tensor2
                        mel_spec, mel_spec2 = pad_to_match(mel_spec, mel_spec2)
                        mel_lengths = mel_lengths2 = mel_lengths if mel_lengths.item() > mel_lengths2.item() else mel_lengths2
                        emotion_pair = (emotion, emotion2)
                        loss, cond, pred, melspec_predicted, noised_base = self.model(
                            mel_spec, text=text_inputs, first_phrase_length=first_phrase_length, emotion=emotion_pair, lens=mel_lengths, noise_scheduler=self.noise_scheduler, masking_type=masking_type, noise_2ndhalf=kwargs['noise_2ndhalf']
                        )
                    else:
                        loss, cond, pred, melspec_predicted, noised_base = self.model(
                            mel_spec, text=text_inputs, first_phrase_length=first_phrase_length, emotion=emotion, lens=mel_lengths, noise_scheduler=self.noise_scheduler, masking_type=masking_type, noise_2ndhalf=kwargs['noise_2ndhalf']
                        )
                    
                    self.accelerator.backward(loss)

                    # Visualize the gradients
                    if kwargs['emotion_conditioning']['emotion_condition_type'] == 'text_mirror':
                        weights_grad = self.model.transformer.input_embed.proj_emotion.weight.grad[:, self.model.transformer.input_embed.proj.weight.shape[1]:]
                        avg_abs_grad_emo = weights_grad.abs().mean()
                        self.accelerator.log({"Mean Abs Grad emotion weights": avg_abs_grad_emo.item()}, step=global_step)
                        weights_grad = self.model.transformer.input_embed.proj_emotion.weight.grad[:, :self.model.transformer.input_embed.proj.weight.shape[1]]
                        avg_abs_grad = weights_grad.abs().mean()
                        self.accelerator.log({"Mean Abs Grad non-emotion weights": avg_abs_grad.item()}, step=global_step)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.is_main:
                    self.ema_model.update()

                global_step += 1

                if self.accelerator.is_local_main_process:
                    self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}, step=global_step)

                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_step)
                        self.writer.add_scalar("lr", self.scheduler.get_last_lr()[0], global_step)
                        # if global_step % 1 == 0:
                        #     self.writer.add_scalar("RAM", get_memory_usage(), global_step)

                progress_bar.set_postfix(step=str(global_step), loss=loss.item())

                if global_step % (kwargs['validation_numsteps'] * self.grad_accumulation_steps) == 0:
                    if kwargs['perform_validation'] and kwargs['validation_numsteps'] != 'every_epoch':
                        with torch.no_grad():
                            asr_device = 'cpu'
                            self._validation(global_step, val_dataloader, nfe_step, cfg_strength, sway_sampling_coef, vocoder, target_sample_rate, faster_whisper_path, asr_device, val_samples_path, kwargs['compute_mcd_valid'], kwargs['compute_wer_valid'], masking_type, kwargs['noise_2ndhalf'])
                    self.save_checkpoint(global_step)

            asr_device = 'cpu'
            if kwargs['perform_validation'] and kwargs['validation_numsteps'] == 'every_epoch':
                with torch.no_grad():
                    self._validation(global_step, val_dataloader, nfe_step, cfg_strength, sway_sampling_coef, vocoder, target_sample_rate, faster_whisper_path, asr_device, val_samples_path, kwargs['compute_mcd_valid'], kwargs['compute_wer_valid'], masking_type, kwargs['noise_2ndhalf'])

        self.save_checkpoint(global_step, last=True)

        self.accelerator.end_training()
