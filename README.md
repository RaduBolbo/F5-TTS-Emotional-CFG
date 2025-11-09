<p align="center">
  <img src="assets/logo.png" alt="F5-TTS-Emotion-CFG Logo" width="500"/>
</p>

# F5-TTS-Emotion-CFG

This repository is based on [**F5-TTS**](https://github.com/SWivid/F5-TTS?tab=readme-ov-file#f5-tts-a-fairytaler-that-fakes-fluent-and-faithful-speech-with-flow-matching): A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching from [paper](https://arxiv.org/abs/2410.06885).

F5-TTS-Emotion-CFG introudces explicit emotion conditioning in F5-TTS zero-shot voice cloning model, by fine-tuning on [ESD](https://arxiv.org/abs/2010.14794) dataset.

The following emotions are supported: Neutral, Happy, Sad, Angry and Surprised.

## 📄 Paper & Citation

Our paper *“Adding Emotion Conditioning in Speech Synthesis via Multi-Term Classifier-Free Guidance”*  
has been accepted at **SpeD 2025**.  

The official BibTeX entry will be added here once the paper is published in IEEE Xplore.  

[![Read Paper](TO DE CONTINUED - INSERT LINK HERE)

Cite the paper as:

  @inproceedings{bolborici2025emotion,
    author    = {Radu-George Bolborici and Ana Antonia Nicolae},
    title     = {Adding Emotion Conditioning in Speech Synthesis via Multi-Term Classifier-Free Guidance},
    booktitle = {Proceedings of the 13th Conference on Speech Technology and Human–Computer Dialogue (SPED 2025)},
    year      = {2025},
    address   = {Cluj-Napoca, Romania},
    note      = {Presented at SPED 2025. Pending indexing in IEEE Xplore.}
  }


## 🚀 Demo

🎧 Explore audio samples generated with the F5-TTS-Emotion-CFG model:

[![Open Demo](https://img.shields.io/badge/Demo-Available-blue?style=for-the-badge)](https://radubolbo.github.io/Conditional-Emotional-F5TTS/) 


## How to install

Step1:

    conda create --name f5-tts-emo python==3.10.0

    conda activate f5-tts-emo

Step 2:

    pip install -e .

## Download models

Execute the script:

    python download_models.py

Or download the models from [🤗 Hugging Face](https://huggingface.co/RaduBolbo/F5-TTS-Emotion-CFG-1/tree/main) and coppy them in the `ckpts` directory.

## How to use (inference)

*⚠️ First of all install the [requirements](#how-to-install) and [download](#download-models) the models.*


Use the CLI interface defiend in `src/f5_tts/infer/infer_cli_emotion.py`:

    python src/f5_tts/infer/infer_cli_emotion.py \
        --ref-audio-path "data/0011_angry.wav" \
        --ref-text "The nine, the eggs, I keep." \
        --inference-text "Hello, this is a text to check emotion." \
        --inference-emotion Surprise \
        --cfg-strength2 10 \
        --output-path "data/output.wav"

- --ref-audio-path: Path to the reference audio file for voice cloning. Provides the speaker’s voice.

- --ref-text: Transcription of the reference audio (the text that is being said).

- --ref-emotion: The emotion in the reference audio clip (if you don't know the reference emotion, in most practical cases it can work with `Neutral` as default, but sometimes it can cause lower voice similarity)

- --inference-text: The new text you want the model to synthesize.

- --inference-emotion: Target emotion for synthesis. Options: `Angry`, `Happy`, `Sad`, `Neutral`, `Surprise`.

- --cfg-strength2: Classifier-free guidance strength for emotion control. Higher = stronger emotion, but too high may reduce naturalness. Typical values range from 2 up to 20.

- --output-path: Path where the generated audio will be saved.
