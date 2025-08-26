import os
import json
from tqdm import tqdm

def create_ravdess_metadata(root, output_path):

    dataset = {"RAVDESS": []}
    emotions_map = {
        "02": "Calm",
        "03": "Happy",
        "04": "Sad",
        "05": "Angry",
        "06": "Fearful",
        "07": "Disgust",
        "08": "Surprise"
    }
    
    phrases_map = {
        "01": "Kids are talking by the door.",
        "02": "Dogs are sitting by the door."
    }

    for root_dir, _, files in os.walk(root):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_parts = file.split("-")

                if len(file_parts) != 7:
                    print(f"Skipping {file}, incorrect format")
                    continue

                _, _, emotion_code, intensity_code, statement_code, _, actor_id = file_parts
                
                emotion = emotions_map.get(emotion_code, "Unknown")
                text = phrases_map.get(statement_code, "Unknown")
                speaker_id = f"Actor_{actor_id[:-4]}"  

                if intensity_code == "02": 
                    dataset["RAVDESS"].append({
                        "phrase_idx": statement_code,
                        "audio_path": os.path.join(root_dir, file),
                        "text": text,
                        "speaker_id": speaker_id,
                        "emotion": emotion,
                        "text_alignment": []  
                    })

    with open(output_path, "w") as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    ravdess_root = "./dataset/RAVDESS/archive/" 
    output_metadata = "./dataset/RAVDESS/ravdess_metadata.json"
    create_ravdess_metadata(ravdess_root, output_metadata)
