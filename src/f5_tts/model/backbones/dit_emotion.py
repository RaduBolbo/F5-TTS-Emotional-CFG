"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
#import torchshow as ts

from x_transformers.x_transformers import RotaryEmbedding

from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

import os
import torch
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="ckpts/runs/emotion_embeddings")

EMBEDDINGS_FILE = "ckpts/emotion_embeddings.pt"

def get_iteration():
    return int(os.environ.get("ITERATION_COUNT", "0"))

def increment_iteration():
    os.environ["ITERATION_COUNT"] = str(get_iteration() + 1)

def get_emotion_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        return torch.load(EMBEDDINGS_FILE)
    return []

def save_emotion_embeddings(embeddings):
    torch.save(embeddings, EMBEDDINGS_FILE)

def register_emotion_embedding(emotion_embed):
    embeddings = get_emotion_embeddings()

    # Clone and move to CPU
    cloned_embedding = emotion_embed.clone().detach().cpu()

    embeddings.append(cloned_embedding)

    # Store up to 100 embeddings
    # if len(embeddings) < 100:
    #     embeddings.append(cloned_embedding)
    # else:
    #     embeddings.pop(0)  # Remove the oldest embedding
    #     embeddings.append(cloned_embedding)

    save_emotion_embeddings(embeddings)

    if len(embeddings) == 200:
        stacked_embeddings = torch.stack(embeddings) 
        writer.add_embedding(stacked_embeddings, global_step=get_iteration(), tag="Emotion Embeddings")
        print(f"Logged {len(embeddings)} emotion embeddings at iteration {get_iteration()}")

# Text embedding
class TextEmbedding(nn.Module):
    def __init__(self, emotion_condition_type, text_num_embeds, text_dim, conv_layers=0, conv_mult=2, emotion_num_embeds=6, emotion_dim=None):
        super().__init__()
        self.emotion_condition_type = emotion_condition_type
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if self.emotion_condition_type == 'text_early_fusion':
            emotion_dim = text_dim
            self.emotion_embeder = nn.Embedding(emotion_num_embeds + 1, emotion_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], emotion, seq_len, drop_text=False, drop_emotion=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx() 
        text = text[:, :seq_len]  # Truncate if text tokens exceed mel spec length (rare case, but included for safety)
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0) # padd to match the cond (melspec) along seq_len.

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d # every token has an embedding, but the last ones will have the 0 filler embeddign from the padding

        if self.emotion_condition_type == 'text_early_fusion':
            # 1) prepare the emotion embeddings
            emotion = emotion + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx() 
            emotion = emotion[:, :seq_len]  # Truncate if text tokens exceed mel spec length (rare case, but included for safety)
            batch, emotion_len = emotion.shape[0], emotion.shape[1]
            emotion = F.pad(emotion, (0, seq_len - emotion_len), value=0) # padd to match the cond (melspec) along seq_len.

            if drop_emotion:  # cfg for emotion
                emotion = torch.zeros_like(emotion)

            emotion = self.emotion_embeder(emotion)  # b n -> b n d # every token has an embedding, but the last ones will have the 0 filler embeddign from the padding

            # 2) add them to text
            text += 0.10 * emotion # Note: The 0.1 weighting factor is heuristic. Intended to preserve learned text embeddings while incorporating emotion.

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)


        return text


# V2) New emotion encoding
class EmotionEmbedding(nn.Module):
    def __init__(self, emotion_num_embeds, emotion_dim, conv_layers=0, conv_mult=2, visualize_emotion_embeddings=False):
        super().__init__()
        self.emotion_embeder = nn.Embedding(emotion_num_embeds + 1, emotion_dim)  # use 0 as filler token
        self.visualize_emotion_embeddings = visualize_emotion_embeddings

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(emotion_dim, self.precompute_max_pos), persistent=False)
            self.emotion_blocks = nn.Sequential(
                *[ConvNeXtV2Block(emotion_dim, emotion_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, emotion: int["b nt"], seq_len, drop_emotion=False):  # noqa: F722
        emotion = emotion + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx() 
        emotion = emotion[:, :seq_len]  # Truncate if text tokens exceed mel spec length (rare case, but included for safety)
        batch, emotion_len = emotion.shape[0], emotion.shape[1]
        emotion = F.pad(emotion, (0, seq_len - emotion_len), value=0) # padd to match the cond (melspec) along seq_len.

        if drop_emotion:  # cfg for emotion
            emotion = torch.zeros_like(emotion)

        emotion = self.emotion_embeder(emotion)  # b n -> b n d # every token has an embedding, but the last ones will have the 0 filler embeddign from the padding

        if self.visualize_emotion_embeddings:
            increment_iteration()
            it = get_iteration()
            print('iteration: ', it)
            if it > 0:
                register_emotion_embedding(emotion[0, 0, :])

            if it > 800:
                writer.flush()

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            emotion_pos_embed = self.freqs_cis[pos_idx]
            emotion = emotion + emotion_pos_embed

            # convnextv2 blocks
            emotion = self.emotion_blocks(emotion)

        return emotion



# noised input audio and context mixing embedding

# V1)
# The old InputEmbedding that has no emotion (🧊)
# class InputEmbedding(nn.Module):
#     def __init__(self, mel_dim, text_dim, emotion_dim, out_dim):
#         super().__init__()
#         self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
#         self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

#     def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
#         if drop_audio_cond:  # cfg for cond audio
#             cond = torch.zeros_like(cond)
#         x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
#         x = self.conv_pos_embed(x) + x
#         return x


# V2) With emotion
class InputEmbedding(nn.Module): # The InputEmeddign with emotion
    def __init__(self, mel_dim, text_dim, emotion_dim, out_dim, emotion_condition_type, load_emotion_weights=True):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim) # old (no emotion)
        self.proj_emotion = nn.Linear(mel_dim * 2 + text_dim + emotion_dim, out_dim) # new (emotion)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        self.weights_setup = False
        self.emotion_condition_type = emotion_condition_type
        self.load_emotion_weights = load_emotion_weights
        
    def _initialize_proj_emotion_weights(self, init_type, weight_reduction_scale=1):
        '''
        Loads the old pretrained weights in the part of proj_emotion that corresponds to x, cond, text_embed and zeros the weights that care used for emotion_embed
        '''
        print("Initializing the InputEmbedding weights to support emotion conditioining")
        torch.set_printoptions(threshold=10_000, linewidth=200)

        with torch.no_grad():

            self.proj_emotion.weight[:, :self.proj.weight.shape[1]] = self.proj.weight  # copy weights for the shared inputs

            if init_type == 'zeros':
                print('Weights initialized as zeros')
                self.proj_emotion.weight[:, self.proj.weight.shape[1]:] = 0 
            elif init_type == 'xavier_reduced':
                print('Weights initialized using Xavier')
                self.proj_emotion.weight[:, self.proj.weight.shape[1]:] = weight_reduction_scale * torch.nn.init.xavier_uniform_(torch.empty_like(self.proj_emotion.weight[:, self.proj.weight.shape[1]:]))

            if self.proj.bias is not None:
                self.proj_emotion.bias = torch.nn.Parameter(self.proj.bias.clone()) # Each output neuron has its own bias, so as output size doesn't get modified this doesn't need to change
        self.weights_setup = True

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], emotion_embed=None, drop_audio_cond=False):  # noqa: F722
        #self._initialize_proj_emotion_weights()
        if not self.weights_setup and self.emotion_condition_type == 'text_mirror':
            if self.load_emotion_weights: # if load_emotion_weights is not requersted, than the weights are expected to be in the model checkpoint and it is not enforced to manually initialize it
                raise RuntimeError("The InputEmbedding instance's weights have not been properly initialized. Please call the _initialize_proj_emotion_weights() method before using it, and make sure to call it only after the other weights have been initialized")
        
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        if self.emotion_condition_type in ['no_emotion_condition', 'text_early_fusion', 'cross_attention']:
            x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        elif self.emotion_condition_type == 'text_mirror':
            x = self.proj_emotion(torch.cat((x, cond, text_embed, emotion_embed), dim=-1))
        else:
            raise NotImplementedError(f'emotion_condition_type {self.emotion_conditioning["emotion_condition_type"]} is not implemented yet')

        x = self.conv_pos_embed(x) + x

        return x

class EmotionCrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, hidden_states, emotion_embedding):
        # Q = model hidden states -> the main source of information
        # K and V = emotion embeddings -> the external information being injected
        attn_output, _ = self.attn(hidden_states, emotion_embedding, emotion_embedding)
        # V1) residual connection
        return hidden_states + attn_output
        # V2)
        #return attn_output # 
        # v3)
        #alpha = 0.2
        #alpha = self.gate(hidden_states)
        #return (1 - alpha) * hidden_states + alpha * attn_output

# Transformer backbone using DiT blocks
class DiTConditioned(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        emotion_num_embeds=6,
        text_dim=None,
        emotion_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        emotion_conditioning={}, # dict associated with emotion conditioning
    ):
        super().__init__()

        self.emotion_conditioning = emotion_conditioning

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
            emotion_dim = mel_dim
        self.text_embed = TextEmbedding(self.emotion_conditioning['emotion_condition_type'], text_num_embeds, text_dim, conv_layers=conv_layers, emotion_num_embeds=emotion_num_embeds, emotion_dim=emotion_dim)
        self.emotion_embed = EmotionEmbedding(emotion_num_embeds, emotion_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, emotion_dim, dim, emotion_condition_type=self.emotion_conditioning['emotion_condition_type'], load_emotion_weights=self.emotion_conditioning['load_emotion_weights'])
        
        # V1) and V2) -> one single corss attention isntance
        self.emotion_cross_attn_blocks = nn.ModuleList([
            EmotionCrossAttention(dim) for _ in range(depth)
        ])

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        emotion,
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        drop_emotion_cond,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time) # construct time (which is in [0, 1]) embedding

        text_embed = self.text_embed(text, emotion, seq_len, drop_text=drop_text, drop_emotion=drop_emotion_cond)
        
        if self.emotion_conditioning['emotion_condition_type'] in ['no_emotion_condition', 'text_early_fusion']:
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond) # no emotion_embed transmitted
        elif self.emotion_conditioning['emotion_condition_type'] == 'cross_attention':
            emotion_embed = self.emotion_embed(emotion, seq_len, drop_emotion=drop_emotion_cond) # still compute emotion embedding
            x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        elif self.emotion_conditioning['emotion_condition_type'] == 'text_mirror':
            emotion_embed = self.emotion_embed(emotion, seq_len, drop_emotion=drop_emotion_cond)
            x = self.input_embed(x, cond, text_embed, emotion_embed, drop_audio_cond=drop_audio_cond)
        else:
            raise NotImplementedError(f'emotion_condition_type {self.emotion_conditioning["emotion_condition_type"]} is not implemented yet')


        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for i, block in enumerate(self.transformer_blocks): 
            if self.emotion_conditioning['emotion_condition_type'] == 'cross_attention':
                # V1) apply coss attention here
                #x = self.emotion_cross_attn(x, emotion_embed)
                # V2) only apply cross attention in the 1st iteration
                # if i==0:
                #     x = self.emotion_cross_attn(x, emotion_embed)
                # V3) One cross attention for each transition
                x = self.emotion_cross_attn_blocks[i](x, emotion_embed)
                 
            x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
