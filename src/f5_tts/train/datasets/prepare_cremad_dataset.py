import os
import json
from tqdm import tqdm
from collections import defaultdict

def create_cremad_metadata(root, output_path):

    dataset = {"CREMA-D": []}

    phrases_map = {
        "DFG": "Don't forget a jacket. ",
        "IEO": "It's eleven o'clock.  ",
        "IOM": "I'm on my way to the meeting. ",
        "ITH": "I think I have a doctor's appointment. ",
        "ITS": "I think I've seen this before. ",
        "IWL": "I would like a new alarm clock. ",
        "IWW": "I wonder what this is about. ",
        "MTI": "Maybe tomorrow it will be cold. ",
        "TAI": "The airplane is almost full. ",
        "TIE": "That is exactly what happened. ",
        "TSI": "The surface is slick. ",
        "WSI": "We'll stop in a couple of minutes. "
    }

    emotions_map = {
        "ANG": "Angry",
        "DIS": "Disgust",
        "FEA": "Fear",
        "SAD": "Sad",
        "HAP": "Happy",
        "NEU": "Neutral"
    }

    intensity_priority = ["HI", "MD", "XX"]

    file_groups = defaultdict(dict)

    for file in os.listdir(root):
        if file.endswith(".wav"):
            file_parts = file.split("_")
            if len(file_parts) != 4:
                print(f"Skipping {file}, incorrect format")
                continue

            speaker_id, phrase_code, emotion_code, intensity_code = file_parts
            intensity_code = intensity_code.replace(".wav", "")

            if phrase_code not in phrases_map or emotion_code not in emotions_map:
                print(f"Skipping {file}, unknown phrase or emotion: {file}")
                continue

            key = (speaker_id, phrase_code, emotion_code)

            if key not in file_groups:
                file_groups[key] = {}
            file_groups[key][intensity_code] = os.path.join(root, file)

    for (speaker_id, phrase_code, emotion_code), intensity_dict in tqdm(file_groups.items()):
        selected_intensity = next((i for i in intensity_priority if i in intensity_dict), None)

        if selected_intensity:
            dataset["CREMA-D"].append({
                "phrase_idx": phrase_code,
                "audio_path": intensity_dict[selected_intensity],
                "text": phrases_map[phrase_code],
                "speaker_id": f"Speaker_{speaker_id}",
                "emotion": emotions_map[emotion_code],
                "intensity": selected_intensity,
                "text_alignment": []
            })

    with open(output_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    cremad_root = "./dataset/CREMA-D/AudioWAV_clean" 
    output_metadata = "./dataset/CREMA-D/cremad_metadata.json"
    create_cremad_metadata(cremad_root, output_metadata)
