import wave
import json
import random
from importlib.resources import files

import torch
import torch.nn.functional as F
import torchaudio
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default


class HFDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        target_sample_rate=24_000,
        n_mel_channels=100,
        hop_length=256,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
    ):
        self.data = hf_dataset
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length

        self.mel_spectrogram = MelSpec(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        )

    def get_frame_len(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]
        sample_rate = row["audio"]["sampling_rate"]
        return audio.shape[-1] / sample_rate * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio = row["audio"]["array"]

        # logger.info(f"Audio shape: {audio.shape}")

        sample_rate = row["audio"]["sampling_rate"]
        duration = audio.shape[-1] / sample_rate

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        audio_tensor = torch.from_numpy(audio).float()

        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.unsqueeze(0)  # 't -> 1 t')

        mel_spec = self.mel_spectrogram(audio_tensor)

        mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t'

        text = row["text"]

        return dict(
            mel_spec=mel_spec,
            text=text,
        )


class CustomDataset(Dataset):
    def __init__(
        self,
        custom_dataset: Dataset,
        durations=None,
        target_sample_rate=24_000,
        hop_length=256,
        n_mel_channels=100,
        n_fft=1024,
        win_length=1024,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = custom_dataset
        self.durations = durations
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    n_mel_channels=n_mel_channels,
                    target_sample_rate=target_sample_rate,
                    mel_spec_type=mel_spec_type,
                ),
            )

    def get_frame_len(self, index):
        if (
            self.durations is not None
        ):  # Please make sure the separately provided durations are correct, otherwise 99.99% OOM
            return self.durations[index] * self.target_sample_rate / self.hop_length
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        audio_path = row["audio_path"]
        text = row["text"]
        duration = float(row["duration"])

        if self.preprocessed_mel:
            mel_spec = torch.tensor(row["mel_spec"])

        else:
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if duration > 30 or duration < 0.3:
                return self.__getitem__((index + 1) % len(self.data))

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            mel_spec = self.mel_spectrogram(audio)
            mel_spec = mel_spec.squeeze(0)  # '1 d t -> d t')

        return dict(
            mel_spec=mel_spec,
            text=text,
        )

def get_audio_duration(audio_path):
    with wave.open(audio_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

class CustomDatasetConditioned(Dataset):
    def __init__(
        self,
        dataset_metadata_path: str,
        #durations=None,
        # target_sample_rate=24_000,
        # hop_length=256,
        # n_mel_channels=100,
        # n_fft=1024,
        # win_length=1024,
        # mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        emotion_conditioning_kwargs: dict = dict(),
    ):
        self.emotion_conditioning_kwargs = emotion_conditioning_kwargs
        self.emotions = self.emotion_conditioning_kwargs['emotions']

        with open(dataset_metadata_path, 'r') as file:
            data = json.load(file)
            if 'ESD' in data:
                data = data['ESD']
            elif 'RAVDESS' in data:
                data = data['RAVDESS']
            elif 'CREMA-D' in data:
                data = data['CREMA-D']
            else:
                raise ValueError('Dataset descriptor has to be either "ESD or "RAVDESS".')

        self.data = self._emotion_filtering(data) # only select allowed emotions
        # create a dict where to keep phrase_idx -> speaker_id -> emotion -> self.data idx (so that I can easily find the index in self.data of a specific phrase of a specific sepaekr and a specific emotion)
        self.data_maping = self.index_data(self.data)
        with open('tmp_dict.json', 'w') as file:
            json.dump(self.data_maping, file)
        #self.durations = durations
        

        self.target_sample_rate = mel_spec_kwargs['target_sample_rate']
        self.hop_length = mel_spec_kwargs['hop_length']
        self.n_fft = mel_spec_kwargs['n_fft']
        self.win_length = mel_spec_kwargs['win_length']
        self.mel_spec_type = mel_spec_kwargs['mel_spec_type']
        self.preprocessed_mel = preprocessed_mel
        self.n_mel_channels = mel_spec_kwargs['n_mel_channels']
        #self.change_emotion_probability = change_emotion_probability
        #self.emotions = {"Angry", "Neutral", "Sad", "Surprise", "Happy"}
        
        self.phrase_idxs = {phrase_idx for phrase_idx in self.data_maping} # a set wit all phrase indexes

        if not self.preprocessed_mel: # if so melspec is computed on the fly in __getitem__()
            self.mel_spectrogram = default(
                mel_spec_module,
                MelSpec(
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    n_mel_channels=self.n_mel_channels,
                    target_sample_rate=self.target_sample_rate,
                    mel_spec_type=self.mel_spec_type,
                ),
            )

    def _emotion_filtering(self, data):
        '''Only allows the emotions that are present in self.emotions'''
        return [sample for sample in data if sample['emotion'] in self.emotions]

    def index_data(self, data):
        '''
        Trasnforms the dir such that any data sampel index can be foudn in linear time if it si searched by phrase_idx, speaker_id and emotion
        '''
        nested_dict = {}

        for index, sample in enumerate(data):
            phrase_idx = sample['phrase_idx']
            speaker_id = sample['speaker_id']
            emotion = sample['emotion']

            if phrase_idx not in nested_dict:
                nested_dict[phrase_idx] = {}
            if speaker_id not in nested_dict[phrase_idx]:
                nested_dict[phrase_idx][speaker_id] = {}
            if emotion not in nested_dict[phrase_idx][speaker_id]:
                nested_dict[phrase_idx][speaker_id][emotion] = []

            nested_dict[phrase_idx][speaker_id][emotion].append(index)
        return nested_dict

    def __len__(self):
        return len(self.data)

    def _sample_2nd_sentence(self, change_emotion, row, emotion, index, first_mel_spec, second_phrase_idx=None):
        '''
        second_phrase_idx can be imposed or it can be randomly selected, if the argument is None
        '''
        #### mix with the second piece of utterance, if needed
        if change_emotion:
            # choose a second utterance
            #emotion_candidates = self.emotions - {emotion}
            if second_phrase_idx is None: # if th e2nd phrase index is not specified
                if self.emotion_conditioning_kwargs['same_sentence']: 
                    second_phrase_idx = row['phrase_idx'] 
                else:
                    second_phrase_idx = random.choice(list(self.phrase_idxs))
            
            #second_emotion = random.choice(list(emotion_candidates))
            try:
                second_emotion = random.choice(list({emotion for emotion in self.data_maping[second_phrase_idx][row["speaker_id"]]} - {emotion})) # a list of all available emotions
                second_row_index = self.data_maping[second_phrase_idx][row["speaker_id"]][second_emotion]
                second_row = self.data[second_row_index[0]]
            except:
                print('!!!! Conditions not met 1. The 1st phrase will be replicated')
                second_row = row
        else:
            attempts = 0
            max_num_attempts = 10
            while attempts < max_num_attempts: # choose a phrase until it has that emotion
                attempts += 1
                if second_phrase_idx is None: # if th e2nd phrase index is not specified
                    if self.emotion_conditioning_kwargs['same_sentence']:  
                        second_phrase_idx = row['phrase_idx']
                    else:
                        second_phrase_idx = random.choice(list(self.phrase_idxs))

                if row["speaker_id"] in self.data_maping[second_phrase_idx]:
                    if emotion in self.data_maping[second_phrase_idx][row["speaker_id"]]:
                        second_row_index = self.data_maping[second_phrase_idx][row["speaker_id"]][emotion]
                        second_row = self.data[second_row_index[0]]
                        break # break only if it was found
            else: # executed if the phrase was not found (break)
                print('!!!! Conditions not met 2. The 1st phrase will be replicated')
                second_row = row

        # load the second piece of utterance
        audio, source_sample_rate = torchaudio.load(second_row["audio_path"])
        duration = get_audio_duration(second_row["audio_path"])

        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        if duration > 30 or duration < 0.3:
            return self.__getitem__((index + 1) % len(self.data))

        if source_sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
            audio = resampler(audio)

        second_mel_spec = self.mel_spectrogram(audio)
        second_mel_spec = second_mel_spec.squeeze(0)  # '1 d t -> d t')

        # concatenate melspecs
        mel_specs = [first_mel_spec, second_mel_spec]

        # concatenate texts and add emotion
        emotions = [row["emotion"], second_row["emotion"]]
        texts = [row["text"], second_row["text"]]

        first_phrase_length = len(texts[0])

        texts_concat = texts[0] + texts[1]
        mel_specs_concat = torch.cat((mel_specs[0], mel_specs[1]), dim=1)

        return mel_specs_concat, texts_concat, emotions, first_phrase_length, second_phrase_idx

    def __getitem__(self, index):
        while True: # loop in case of 'contrastive_loss' tha tdoesn't find  a propper configuration
            row = self.data[index]

            audio_path = row["audio_path"]
            text = row["text"] 
            text_alignment = row["text_alignment"]
            emotion = row["emotion"]

            duration = get_audio_duration(audio_path)

            #### load the melspectrogram of the base utterance
            audio, source_sample_rate = torchaudio.load(audio_path)
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)

            if duration > 30 or duration < 0.3:
                return self.__getitem__((index + 1) % len(self.data))

            if source_sample_rate != self.target_sample_rate:
                resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
                audio = resampler(audio)

            first_mel_spec = self.mel_spectrogram(audio)
            first_mel_spec = first_mel_spec.squeeze(0)  # '1 d t -> d t')



            if self.emotion_conditioning_kwargs['contrastive_loss']:
                mel_specs_concat_changed, texts_concat_changed, emotions_changed, first_phrase_length_changed, second_phrase_idx = self._sample_2nd_sentence(True, row, emotion, index, first_mel_spec)
                mel_specs_concat_unchanged, texts_concat_unchanged, emotions_unchanged, first_phrase_length_unchanged, _ = self._sample_2nd_sentence(False, row, emotion, index, first_mel_spec, second_phrase_idx=second_phrase_idx)
                if texts_concat_changed != texts_concat_unchanged or emotions_changed == emotions_unchanged:
                    index += 1 # skip this sample
                    break
                else:
                    sample = [
                        dict(
                            mel_spec=mel_specs_concat_changed,
                            text=texts_concat_changed,
                            emotion=emotions_changed,
                            first_phrase_length=first_phrase_length_changed,
                        ),
                        dict(
                            mel_spec=mel_specs_concat_unchanged,
                            text=texts_concat_unchanged,
                            emotion=emotions_unchanged,
                            first_phrase_length=first_phrase_length_unchanged,
                        ),
                    ]
                    return sample
            else:
                # else: only one utterance which can have the same emotion or not
                change_emotion = random.uniform(0, 1) < self.emotion_conditioning_kwargs['change_emotion_probability']
                mel_specs_concat, texts_concat, emotions, first_phrase_length, _ = self._sample_2nd_sentence(change_emotion, row, emotion, index, first_mel_spec)
                return [dict(
                    mel_spec=mel_specs_concat,
                    text=texts_concat,
                    emotion=emotions,
                    first_phrase_length=first_phrase_length,
                )]
                # return dict(
                #     mel_spec=mel_specs_concat,
                #     text=texts_concat,
                #     emotion=emotions,
                #     first_phrase_length=first_phrase_length,
                # )

# Dynamic Batch Sampler


class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_last: bool = False
    ):
        print('frames_threshold: ', frames_threshold)
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 0:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_last and len(batch) > 0:
            batches.append(batch)
        

        del indices

        # if want to have different batches between epochs, may just set a seed and log it in ckpt
        # cuz during multi-gpu training, although the batch on per gpu not change between epochs, the formed general minibatch is different
        # e.g. for epoch n, use (random_seed + n)
        random.seed(random_seed)
        random.shuffle(batches)

        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_name: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
    emotion_conditioning_kwargs: dict = dict(),
) -> CustomDataset | HFDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """

    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        #rel_data_path = str(files("f5_tts").joinpath(f"../../data/{dataset_name}_{tokenizer}"))
        #rel_data_path = str(files("f5_tts").joinpath(f"data/{dataset_name}_{tokenizer}"))
        rel_data_path = f"data/{dataset_name}_{tokenizer}"
        if audio_type == "raw":
            try:
                train_dataset = load_from_disk(f"{rel_data_path}/raw")
            except:  # noqa: E722
                train_dataset = Dataset_.from_file(f"{rel_data_path}/raw.arrow")
            preprocessed_mel = False
        elif audio_type == "mel":
            train_dataset = Dataset_.from_file(f"{rel_data_path}/mel.arrow")
            preprocessed_mel = True
        with open(f"{rel_data_path}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset,
            durations=durations,
            preprocessed_mel=preprocessed_mel,
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )

    elif dataset_type == "CustomDatasetPath":
        try:
            train_dataset = load_from_disk(f"{dataset_name}/raw")
        except:  # noqa: E722
            train_dataset = Dataset_.from_file(f"{dataset_name}/raw.arrow")

        with open(f"{dataset_name}/duration.json", "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        durations = data_dict["duration"]
        train_dataset = CustomDataset(
            train_dataset, durations=durations, preprocessed_mel=preprocessed_mel, **mel_spec_kwargs
        )

    elif dataset_type == "HFDataset":
        print(
            "Should manually modify the path of huggingface dataset to your need.\n"
            + "May also the corresponding script cuz different dataset may have different format."
        )
        pre, post = dataset_name.split("_")
        train_dataset = HFDataset(
            load_dataset(f"{pre}/{pre}", split=f"train.{post}", cache_dir=str(files("f5_tts").joinpath("../../data"))),
        )

    elif dataset_type == "CustomDatasetConditioned":
        # In this situation (emotion dataset) dataset_name should be a path to a directory with the approapriate structure
        train_dataset = CustomDatasetConditioned(dataset_name, mel_spec_kwargs=mel_spec_kwargs, emotion_conditioning_kwargs=emotion_conditioning_kwargs)

    return train_dataset


# collation

def collate_fn_emotion(batch):
    dicts = []
    for example_idx in range(len(batch[0])):
        mel_specs = [item[example_idx]["mel_spec"].squeeze(0) for item in batch]
        mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
        max_mel_length = mel_lengths.amax()

        padded_mel_specs = []
        for spec in mel_specs:  # TODO. maybe records mask for attention here
            padding = (0, max_mel_length - spec.size(-1))
            padded_spec = F.pad(spec, padding, value=0)
            padded_mel_specs.append(padded_spec)

        mel_specs = torch.stack(padded_mel_specs)

        text = [item[example_idx]["text"] for item in batch]
        emotion = [item[example_idx]["emotion"] for item in batch]
        text_lengths = torch.LongTensor([len(item) for item in text])
        first_phrase_length = [item[example_idx]["first_phrase_length"] for item in batch]

        dicts.append(dict(
            mel=mel_specs,
            mel_lengths=mel_lengths,
            text=text,
            emotion=emotion,
            text_lengths=text_lengths,
            first_phrase_length=first_phrase_length
        ))
    
    return dicts

def collate_fn(batch):
    mel_specs = [item["mel_spec"].squeeze(0) for item in batch]
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()

    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    text_lengths = torch.LongTensor([len(item) for item in text])

    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
    )
