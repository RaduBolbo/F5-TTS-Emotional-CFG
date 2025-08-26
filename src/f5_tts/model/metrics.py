import numpy as np
from nnmnkwii.metrics import melcd
from fastdtw import fastdtw
from pymcd.mcd import Calculate_MCD
from scipy.spatial.distance import euclidean
import scipy
from faster_whisper import WhisperModel
import os
import torch
import traceback
import itertools
import re
from num2words import num2words


__MODEL_HASH = 'f0fe81560cb8b68660e564f55dd99207059c092e'


def load_model(model_path, language, precision):

    ### checking the model is present
    model_bin_path = os.path.join(model_path, 'snapshots', __MODEL_HASH, 'model.bin')
    model_bin_dir_path = os.path.dirname(model_bin_path)
    if not os.path.isfile(model_bin_path):
        raise Exception(f"[ERROR] Model file {model_bin_path} does not exist ! Please make sure the installation is correct.")   
    ### end check

    if language == 'auto':  # not used now but kept for future updates
        language = None
    
    print(f"[INFO] Loading Faster-Whisper from {model_path} ...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = WhisperModel(model_bin_dir_path, device=device, compute_type=precision)
    print(f"[INFO] Faster-Whisper loaded!")
    
    return model


def execute_asr_faster(input_wav_filepath, model, language, word_timestamps=False):
    try:
        segments, info = model.transcribe(
            audio=input_wav_filepath,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=700),
            language=language,
            word_timestamps=word_timestamps
        )
        text = ''

        words = []
        if word_timestamps:
            for segment in segments:
                for word_info in segment.words:
                    words.append([word_info.word, word_info.start, word_info.end])
        words = [
            [word.strip().strip('-').strip('.').strip(',').strip(';').strip('!').strip('?').strip('`').strip('"').strip("'").strip(":"), float(start), float(end)]
            for word, start, end in words
        ]

        if text == '':
            for segment in segments:
                text += segment.text
    except:
        return traceback.format_exc()
    
    if word_timestamps:
        return words
    else:
        return text


def get_mcd(mel1, mel2):
    # mel1 = mel1.numpy()  # Shape: (X, 100)
    # mel2 = mel2.numpy()  # Shape: (X, 100)

    mcd_value = melcd(mel1, mel2)
    return mcd_value

def align_with_dtw(X, Y):
    _, path = fastdtw(X, Y, dist=lambda x, y: np.linalg.norm(x - y))
    aligned_X = np.array([X[i] for i, _ in path])
    aligned_Y = np.array([Y[j] for _, j in path])
    return aligned_X, aligned_Y

class Calculate_MCD_from_ndarray(Calculate_MCD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def average_mcd(self, ref_mcep_audio, syn_mcep_audio, cost_function, MCD_mode, audios_sample_rate):
        """
        Calculate the average MCD.
        :param ref_mcep_files: list of strings, paths to MCEP target reference files
        :param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
        :param cost_function: distance metric used
        :param plain: if plain=True, use Dynamic Time Warping (dtw)
        :returns: average MCD, total frames processed
        """

        # resample
        ref_mcep_audio = scipy.signal.resample(ref_mcep_audio, int(len(ref_mcep_audio) * self.SAMPLING_RATE / audios_sample_rate))
        syn_mcep_audio = scipy.signal.resample(syn_mcep_audio, int(len(syn_mcep_audio) * self.SAMPLING_RATE / audios_sample_rate))

        # extract MCEP features (vectors): 2D matrix (num x mcep_size)
        ref_mcep_vec = self.wav2mcep_numpy(ref_mcep_audio)
        syn_mcep_vec = self.wav2mcep_numpy(syn_mcep_audio)

        if MCD_mode == "plain":
            path = []
            for i in range(len(ref_mcep_vec)):
                path.append((i, i))
        elif MCD_mode == "dtw":
            _, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)
        elif MCD_mode == "dtw_sl":
            cof = len(ref_mcep_vec)/len(syn_mcep_vec) if len(ref_mcep_vec)>len(syn_mcep_vec) else len(syn_mcep_vec)/len(ref_mcep_vec)
            _, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)

        frames_tot, min_cost_tot = self.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, path)

        if MCD_mode == "dtw_sl":
            mean_mcd = cof * self.log_spec_dB_const * min_cost_tot / frames_tot
        else:
            mean_mcd = self.log_spec_dB_const * min_cost_tot / frames_tot

        return mean_mcd
    
    def calculate_mcd(self, reference_audio, synthesized_audio, audios_sample_rate):
        # extract acoustic features
        mean_mcd = self.average_mcd(reference_audio, synthesized_audio, self.log_spec_dB_dist, self.MCD_mode, audios_sample_rate)

        return mean_mcd

def get_mcd_dtw(audio1, audio2, audios_sample_rate):
    mcd_toolbox = Calculate_MCD_from_ndarray(MCD_mode="dtw")
    mcd_value = mcd_toolbox.calculate_mcd(audio1, audio2, audios_sample_rate)

    return mcd_value

class WhisperModelAdapter():
    def __init__(self, faster_whisper_path, text_language, precision, device):
        self.__MODEL_HASH = 'f0fe81560cb8b68660e564f55dd99207059c092e'
        if device == 'cpu':
            precision = 'float32'
        self.load_model(faster_whisper_path, text_language, precision, device)

    def load_model(self, model_path, language, precision, device):

        ### checking the model is present
        model_bin_path = os.path.join(model_path, 'snapshots', self.__MODEL_HASH, 'model.bin')
        model_bin_dir_path = os.path.dirname(model_bin_path)
        if not os.path.isfile(model_bin_path):
            raise Exception(f"[ERROR] Model file {model_bin_path} does not exist ! Please make sure the installation is correct.")   
        ### end check

        if language == 'auto':  # not used now but kept for future updates
            language = None
        
        print(f"[INFO] Loading Faster-Whisper from {model_path} ...")
        #device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.model = WhisperModel(model_bin_dir_path, device=device, compute_type=precision)
        print(f"[INFO] Faster-Whisper loaded!")
        
    def execute_asr_faster(self, wav_or_filepath, audios_sample_rate, language):

        if isinstance(wav_or_filepath, np.ndarray):
            target_sr = self.model.feature_extractor.sampling_rate
            wav_or_filepath = scipy.signal.resample(wav_or_filepath, int(len(wav_or_filepath) * target_sr / audios_sample_rate))

        try:
            segments, info = self.model.transcribe(
                audio=wav_or_filepath,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language
            )
            text = ''

            if text == '':
                for segment in segments:
                    text += segment.text
        except:
            return traceback.format_exc()
        
        return text

def replace_special_characters(text, lang):
    characters_to_replace = "、,;`.。？！—!?()~'[]*&^:》《、…・　（）「」『』【】｛｝‥〜ー§＃"
    translation_table = str.maketrans(characters_to_replace, " " * len(characters_to_replace))
    text = text.translate(translation_table)

    text = number_to_literal_in_string(text, lang)

    return text

def number_to_literal_in_string(text, lang='en'):
    pattern = re.compile(r'(\d+)')
    text = pattern.sub(r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    def replace_number(match):
        number = match.group(0)
        try:
            return num2words(float(number), lang=lang)
        except Exception as e:
            return str(e)
    pattern = re.compile(r'\b\d+(\.\d+)?\b')
    result = pattern.sub(replace_number, text)

    return result

def generate_variations(text):
    '''
    Generate variations where each pair of words written with "-" can be written by replaceing the "-" with a whitespace or removed.

    Parameters
    ----------
    text : str
        The first text

    Returns
    -------
    variations : lsit
        A lsit of all possible variations of the phrase, where every pair of words written woith "-" can be splitted or merged.
    '''
    parts = text.split('-')

    combinations = list(itertools.product(['', ' '], repeat=len(parts) - 1))

    variations = set()
    for combo in combinations:
        variation = parts[0]
        for part, separator in zip(parts[1:], combo):
            variation += separator + part
        variations.add(variation)

    return variations

def link_words(phrase1, phrase2):
    words1 = phrase1.split()
    words2 = phrase2.split()

    concatenated_words = {}
    for i in range(len(words1)):
        for j in range(i + 1, len(words1) + 1):
            concatenated_word = ''.join(words1[i:j])
            concatenated_words[concatenated_word] = (i, j)

    linked_phrase = []
    skip_until = -1
    for i, word in enumerate(words1):
        if i <= skip_until:
            continue
        merged = False
        for concat_word, (start, end) in concatenated_words.items():
            if word == words1[start] and concat_word in words2 and i == start:
                linked_phrase.append(concat_word)
                skip_until = end - 1
                merged = True
                break
        if not merged:
            linked_phrase.append(word)

    return ' '.join(linked_phrase)


def get_error(ref, pred, error_fucntion):
    """
    Computes the best error wetween the 2 phrases, considering all variations that can arise from the use of "-"

    Parameters
    ----------
    text1 : str
        The first text
    text2 : str
        The first text
    error_fucntion
        The fucntion used to compute the error. For example: jiwer.wil or jiwer.wer

    Returns
    -------
    min(candidate_error_list) : float
        The best possible error.
    """
    variations_ref = generate_variations(ref)
    variations_pred = generate_variations(pred)
    candidate_error_list = []
    for varaition_ref in variations_ref:
        for variation_pred in variations_pred:
            candidate_error_list.append(error_fucntion(varaition_ref, variation_pred))
    return min(candidate_error_list)