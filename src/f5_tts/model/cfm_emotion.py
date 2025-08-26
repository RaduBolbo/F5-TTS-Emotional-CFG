"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


class CFMConditioned(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask

        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

        # emotion tokens # this should move into a config file
        self.emotion_dict = {
            "Angry": 1,
            "Neutral": 2,
            "Sad": 3,
            "Surprise": 4,
            "Happy": 5
        } # 0 is reserved for filler token

    @property
    def device(self):
        return next(self.parameters()).device

    def tokenize_emotion(self, emotion, text_shape, first_phrase_length,initial_text_dims, device):
        emotion = torch.tensor([[self.emotion_dict.get(emotion, 0) for emotion in batch] for batch in emotion], device=device)
        if 0 in emotion:
            raise('Unknown emotion found in the emotion conditioning input.')
        emotion_tokens = torch.full((emotion.shape[0], text_shape), -1, device=device)
        for i in range(len(first_phrase_length)):
            emotion_tokens[i, first_phrase_length[i]:initial_text_dims[i]] = emotion[i, 1].item()
            emotion_tokens[i, :first_phrase_length[i]] = emotion[i, 0].item()
        return emotion_tokens

    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        first_phrase_length,
        emotion,
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        cfg_strength2=None,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
    ):
        self.eval()
        # raw wave

        initial_text_dims = [len(line) for line in text]

        if cond.ndim == 2:
            cond = self.mel_spec(cond)
            cond = cond.permute(0, 2, 1)
            assert cond.shape[-1] == self.num_channels

        cond = cond.to(next(self.parameters()).dtype)

        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # text
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        if exists(text):
            text_lens = (text != -1).sum(dim=-1)
            lens = torch.maximum(text_lens, lens)

        # Create the emotion_tokens tensor
        emotion = self.tokenize_emotion(emotion, text.shape[1], first_phrase_length, initial_text_dims, device)

        # duration
        cond_mask = lens_to_mask(lens)
        if edit_mask is not None:
            cond_mask = cond_mask & edit_mask

        if isinstance(duration, int):
            duration = torch.full((batch,), duration, device=device, dtype=torch.long)

        duration = torch.maximum(lens + 1, duration)
        duration = duration.clamp(max=max_duration)
        max_duration = duration.amax()

        # duplicate-test branch for internal time-step observation.
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        cond = F.pad(cond, (0, 0, 0, max_duration - cond_seq_len), value=0.0)
        cond_mask = F.pad(cond_mask, (0, max_duration - cond_mask.shape[-1]), value=False)
        cond_mask = cond_mask.unsqueeze(-1)
        # is the reference melspec, concatenated with zeros over the text-generation portion.
        step_cond = torch.where(
            cond_mask, cond, torch.zeros_like(cond)
        )  # allow direct control (cut cond audio) with lens passed in

        if batch > 1:
            mask = lens_to_mask(duration)
        else:  # save memory and speed up, as single inference need no mask currently
            mask = None

        # test for no ref audio
        if no_ref_audio:
            cond = torch.zeros_like(cond)

        # neural ode

        def fn(t, x):
            # predict flow
            pred = self.transformer(
                x=x, cond=step_cond, text=text, emotion=emotion, time=t, mask=mask, drop_audio_cond=False, drop_text=False, drop_emotion_cond=False
            )
            if cfg_strength < 1e-5:
                return pred

            null_pred = self.transformer(
                x=x, cond=step_cond, text=text, emotion=emotion, time=t, mask=mask, drop_audio_cond=True, drop_text=True, drop_emotion_cond=True
            )

            if cfg_strength2:
                half_pred = self.transformer(
                    x=x, cond=step_cond, text=text, emotion=emotion, time=t, mask=mask, drop_audio_cond=False, drop_text=False, drop_emotion_cond=True
                )
                return pred + (pred - null_pred) * cfg_strength + (pred - half_pred) * cfg_strength2
            else:
                print('cfg_strength2 is None. Classical CFM applied')
                
                return pred + (pred - null_pred) * cfg_strength

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(dur, self.num_channels, device=self.device, dtype=step_cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        t_start = 0

        # duplicate test corner for inner time step observation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))

        t = torch.linspace(t_start, 1, steps, device=self.device, dtype=step_cond.dtype) # the sequence of timestamps for sampling

        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t) # alter the sequence of timestamps according to the sway sampling.

        trajectory = odeint(fn, y0, t, **self.odeint_kwargs) # contains all flow steps

        sampled = trajectory[-1] # save the last function evaluation; the last function evaluation corresponds to t=1 where the flow is complete
        out = sampled

        out = torch.where(cond_mask, cond, out)

        if exists(vocoder):
            out = out.permute(0, 2, 1)
            out = vocoder(out)

        return out, trajectory

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        first_phrase_length,
        emotion,
        masking_type = 'original',
        noise_2ndhalf = 'uniform',
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
    ):
        '''
        masking_type:
          - 'original' = A fraction of the batch is sampled from a uniform distribution to determine the percentage of True values in a mask
          - '2nd_part_proportional_masked' = the 2nd part of speech is masked proportional to how long the 1st and 2nd texts are (may not be 100% correct all the times)
        '''
        change_emotion_forward = isinstance(inp, tuple)
        contrastive_loss = isinstance(emotion, tuple)
        if contrastive_loss:
            emotion, emotion2 = emotion
            #print(emotion, emotion2)

        mode = None
        if change_emotion_forward:
            #mode = 'binary_interpolation'
            mode = 'triple_interpolation'
            if mode == 'binary_interpolation':
                change_target = random() < 0.5
                if change_target:
                    mel_target, mel_source = inp
                    inp = mel_source
                else:
                    mel_target, mel_source = inp
                    inp = mel_target
            elif mode == 'triple_interpolation':
                mel1, mel2 = inp
                #mel_source = 
                mel_target = mel1
                inp = mel_target

    
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        initial_text_dims = [len(line) for line in text]

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma

        # handle text as string
        if isinstance(text, list):
            if exists(self.vocab_char_map):
                text = list_str_to_idx(text, self.vocab_char_map).to(device)
            else:
                text = list_str_to_tensor(text).to(device)
            assert text.shape[0] == batch

        # Create the emotion_tokens tensor
        emotion = self.tokenize_emotion(emotion, text.shape[1], first_phrase_length, initial_text_dims, device)
        if contrastive_loss:
            emotion2 = self.tokenize_emotion(emotion2, text.shape[1], first_phrase_length, initial_text_dims, device)
        # print('emotion after torkenizing in CFM (before transfoermer): ')
        # print('emotion.shape: ', emotion.shape)
        # print('emotion: ', emotion)

        # lens and mask
        if not exists(lens): # lens are usually provided from then trainig script
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # get a random span to mask out for training conditionally
        proportional_length = (torch.tensor(first_phrase_length, dtype=torch.float32) / torch.tensor(initial_text_dims, dtype=torch.float32)) * seq_len
        proportional_length = proportional_length.round().long()

        if masking_type == 'original':
            frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
            rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)
        elif masking_type == '2nd_part_proportional_masked':
            rand_span_mask = torch.zeros((batch, seq_len), dtype=torch.bool, device=inp.device)

            for i in range(batch):
                rand_span_mask[i, proportional_length[i]:] = True

        if exists(mask):
            rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        if mode == 'triple_interpolation':
            if random() < 0.5:
                mel_source = x0
                φ = (1 - t) * mel_source + t * x1
            else:
                mel_source = (x0 * mel2)/2
                φ = (1 - t) * mel_source + t * x1
        else:
            φ = (1 - t) * x0 + t * x1
            
        if change_emotion_forward:
            if mode == 'binary_interpolation':
                flow = mel_target - x0
            elif mode == 'triple_interpolation':
                flow = mel_target - mel_source
        else:
            flow = x1 - x0

        # only predict what is within the random mask span for infilling
        cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1) # where the rand_span_mask is True, replace the x1 (target melspec) with zeros => obtain the condition (cond): (1-m) * x1

        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
            drop_emotion_cond = True
        else:
            drop_text = False
            drop_emotion_cond = False

        drop_emotion_cond = False

        # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, emotion=emotion, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_emotion_cond=drop_emotion_cond
        )
        if contrastive_loss:
            # drop emotion
            pred2 = self.transformer(
                x=φ, cond=cond, text=text, emotion=emotion2, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text, drop_emotion_cond=True
            )

        # flow matching loss
        loss = F.mse_loss(pred, flow, reduction="none")
        loss = loss[rand_span_mask]

        if contrastive_loss:
            loss_contrastive = -F.mse_loss(pred2, pred, reduction="none")
            loss_contrastive = loss_contrastive[rand_span_mask]

            c = 2 # Empirical
            loss = loss
            loss_contrastive = loss_contrastive
            constrastive_loss_type = 'bound'
            if constrastive_loss_type == 'unbound':
                loss = loss + c * loss_contrastive
            elif constrastive_loss_type == 'bound':
                loss = loss + c * torch.sigmoid(loss_contrastive)

        melspec_predicted = pred + x0

        return loss.mean(), cond, pred, melspec_predicted, φ
