import os
import pprint
from tqdm import tqdm
from f5_tts.model.metrics import WhisperModel, load_model, execute_asr_faster
from collections import Counter
import subprocess
import json
import shutil
from pathlib import Path
from f5_tts.train.datasets.utils import SuppressOutput

compute_alignment = False
if compute_alignment:
    mfa_root_dir = os.path.join(os.getcwd(), "mfa_root_dir")
    os.environ['MFA_ROOT_DIR'] = mfa_root_dir
    from montreal_forced_aligner.alignment import PretrainedAligner


def perform_mfa_alignment(audio_path, transcription, output_dir, dictionary_path, model_dir):
    """
    Perform forced alignment using Montreal Forced Aligner for a single audio file.

    Args:
        audio_path (str): Path to the audio file.
        transcription (str): Text transcription of the audio.
        output_dir (str): Path to save alignment output.
        dictionary_path (str): Path to MFA's pronunciation dictionary.
        model_dir (str): Path to MFA's pretrained acoustic model directory.

    Returns:
        str: Path to the generated TextGrid or JSON file.
    """
    # Validate audio file existence
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' does not exist.")
        return None

    # Create temporary directory
    temp_dir = "mfa_temp_single"
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare paths
    transcription_file = os.path.join(temp_dir, "temp_transcription.lab")
    audio_temp_path = os.path.join(temp_dir, "temp_audio.wav")

    # Write transcription to a temp file
    with open(transcription_file, "w", encoding="utf8") as f:
        f.write(transcription)

    # Copy audio file to the temp directory
    os.system(f'cp "{audio_path}" "{audio_temp_path}"')


    # Run MFA alignment
    try:
        subprocess.run(
            [
                "mfa", "align",
                temp_dir,          # Temp directory containing audio and transcription
                dictionary_path,   # Pronunciation dictionary
                model_dir,         # Pretrained model
                output_dir,        # Output directory for alignment
                "--clean",         # Remove temporary files
                "--output_format", "json",  # Generate JSON output
                "--verbose",       # Verbose logging
                "--single_speaker"  # Optional: Use for single speaker audio
            ],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error during MFA alignment: {e}")
        return None

    # Get the output file path
    output_file = os.path.join(output_dir, "temp_audio.TextGrid")
    if not os.path.exists(output_file):
        print("Error: Alignment failed. Output file not found.")
        return None

    print(f"Alignment completed successfully. Output saved at: {output_file}")
    return output_file

    
def run_alignment(input_audio_path, tag_free_text, dict_path, model_path, output_file, temp_path_to_delete, corpus_dir_name):
    """
    Aligns text with audio and saves a .json that contains timestamps for each word.
    
    Parameters
    -------
    input_audio_path : str 
        Path to the audio file to align.
    tag_free_text : str
        The audio transcription text without user tags.
    dict_path : str
        Path to the dictionary used by MFA.
    model_path : str
        Path to the acoustic model used by MFA.
    output_file : str
        Path where the output .json will be saved.
    """

    mfa_root_dir = "MFA_TMP"
    cache_to_delete = os.path.join(mfa_root_dir, corpus_dir_name) # this file has to be deleted because of caching reasons
    shutil.rmtree(cache_to_delete) if os.path.isdir(cache_to_delete) else None

    # Step 1: create the corpus
    corpus_dir = os.path.join(temp_path_to_delete, corpus_dir_name)
    os.makedirs(corpus_dir, exist_ok=True)
    
    audio_dest_path = os.path.join(corpus_dir, "file.wav")
    shutil.copy(input_audio_path, audio_dest_path)
    
    text_dest_path = os.path.join(corpus_dir, "file.txt")
    with open(text_dest_path, 'w', encoding = 'utf-8') as file: # This is needed for the corpus alignment
        file.write(tag_free_text)

    corpus_dir, dict_path, model_path = Path(corpus_dir), Path(dict_path), Path(model_path)

    # Step 2: align the corpus

    aligner = PretrainedAligner(
        corpus_directory=corpus_dir,
        dictionary_path=dict_path,
        acoustic_model_path=model_path,
    )

    try:
        with SuppressOutput():
            aligner.align()
            aligner.analyze_alignments()
            aligner.export_files(corpus_dir, output_format="json")
        shutil.copy(os.path.join(corpus_dir, 'file.json'), output_file) # **** at this point some formatting such as eliminating phonemes should be done.  Let's see the desired format
        print("[INFO] Forced alignment performed successfully!")
    except Exception as e:
        print(f"[INFO] Alignment failed: {e}")
        raise
    finally:
        with SuppressOutput():
            aligner.cleanup()


def create_emotion_dataset(root, dataset_metadata_output_path, dataset_type):
    faster_whisper_path = 'ckpts/resources/models/models--Systran--faster-whisper-large-v2-local'
    text_language='en'
    asr_model = load_model(faster_whisper_path, text_language, 'float16')

    if dataset_type == 'ESD':
        speaker_ids = [speaker_id for speaker_id in os.listdir(root) if os.path.isdir(os.path.join(root, speaker_id))]
        emotions = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']
        phrases_dict = {}
        for speaker_id in speaker_ids:
            for emotion in emotions:
                phrase_idx = 0
                for audio_name in sorted(os.listdir(os.path.join(root, speaker_id, emotion))):
                    audio_path = os.path.join(root, speaker_id, emotion, audio_name)
                    if phrase_idx not in phrases_dict:
                        phrases_dict[phrase_idx] = [audio_path]
                    else:
                        phrases_dict[phrase_idx].append(audio_path)
                    phrase_idx += 1

        dataset = {'ESD': []}
        error_idx = 0
        for phrase_idx in tqdm(phrases_dict.keys()):
            # an ASR is needed 
            transcription_texts = []
            for audio_path in phrases_dict[phrase_idx]:
                transcription_text = execute_asr_faster(
                    input_wav_filepath=audio_path,
                    model=asr_model,
                    language=text_language,
                )
                transcription_texts.append(transcription_text)
            transcription_counts = Counter(transcription_texts)
            most_probable_transcription, count = transcription_counts.most_common(1)[0]
            if compute_alignment:
                shutil.rmtree(mfa_root_dir) if os.path.isdir(mfa_root_dir) else None # delete the MFA cache 
            for audio_path in phrases_dict[phrase_idx]:
                if compute_alignment:
                    # uncomment for MFA
                    # model_path = os.path.join('ckpts/resources/models/MFA', f"{text_language}_mfa.zip")
                    # dict_path = os.path.join('ckpts/resources/models/MFA', f"{text_language}_mfa.dict")
                    # dir_path = os.path.join('ckpts/resources/models/MFA', f"{text_language}_mfa.zip") # , dictionary_path='temporary_aligned.json'
                    # output_file = perform_mfa_alignment(audio_path, most_probable_transcription, 'TMP', dictionary_path=dict_path, model_dir=model_path)
                    # audio_temp_path = os.path.join('TMP', "temp_audio.wav")
                    # temp_path_to_delete = "TMP/tmptdl"
                    # corpus_dir_name = "MFA_TMP_CORPUS_DIR"
                    # run_alignment(audio_path, most_probable_transcription, dict_path, model_path, audio_temp_path, temp_path_to_delete, corpus_dir_name) 
                    # alignment = 1 
                    # alignment_path = os.path.join(temp_path_to_delete, corpus_dir_name, 'file.json') # path to the alignment
                    # with open(alignment_path, "r") as file:
                    #     data = json.load(file)
                    #     word_entries = data["tiers"]["words"]["entries"]
                    word_entries = execute_asr_faster(
                        input_wav_filepath=audio_path,
                        model=asr_model,
                        language=text_language,
                        word_timestamps=True
                        )
                else:
                    word_entries = []
                #shutil.rmtree(temp_path_to_delete) if os.path.isdir(temp_path_to_delete) else None # uncomment for MFA

                emotion = os.path.basename(os.path.dirname(audio_path)) # find the emotion from the diectory name
                speaker_id = os.path.basename(os.path.dirname(os.path.dirname(audio_path)))
                data_example = {'phrase_idx': phrase_idx, 'audio_path': audio_path, 'text': most_probable_transcription, 'speaker_id': speaker_id, 'emotion': emotion, 'text_alignment': word_entries} # create the dataset sample
                dataset['ESD'].append(data_example)

        with open(dataset_metadata_output_path, "w") as json_file:
            json.dump(dataset, json_file, indent=4)


if __name__ == '__main__':
    create_emotion_dataset('dataset/ESD/train', 'dataset/ESD/train/dataset_descriptor.json', dataset_type='ESD')
