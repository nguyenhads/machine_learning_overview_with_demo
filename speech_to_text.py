# -*- coding: utf-8 -*-
"""speech_to_text
Author: Nguyen Thai Ha
About:
  * This program demos the using of Whisper OpenAI to transcribe the speech to text

"""

# install Whisper
# pip install git+https://github.com/openai/whisper.git

# importing libs
import os
import time
from pathlib import Path 
from typing import Iterator, TextIO

import torch
import whisper

# define function to convert speech to text
def transcribe_audio(input_audio, model_type, device):
  model = whisper.load_model(model_type, device)
  transcribe_result = model.transcribe(input_audio)
  return transcribe_result

# transcribing
model_type = "large"
device = "cuda" if torch.cuda.is_available() else "cpu"
input_audio_path = "/content/sample_data/example.m4a"
start_time = time.time()
print("Transcribing ... ")
transcribe_result = transcribe_audio(input_audio_path, model_type=model_type, device=device)
print("Finnished !")
end_time = time.time()
print(f"Transcribed time: {(end_time - start_time) / 60 :.2f} min")

# check transcribed results
print(transcribe_result.keys())
print(transcribe_result["text"][:100])

# function to save as txt file
def write_txt(transcript: Iterator[dict], file: TextIO):
    for segment in transcript:
        print(segment['text'].strip(), file=file, flush=True)

# saving file
output_file_name = input_audio_path.split("/")[-1].split(".")[0] + ".txt"
output_file_path = Path(f"/content/drive/MyDrive/Colab Notebooks/01_speech_to_text/result/{output_file_name}")
output_file_path.touch(exist_ok=True)

# checking an output directory
print(output_file_path)

# save file and close
 with open(output_file_path, "w", encoding="utf-8") as txt:
            write_txt(transcribe_result["segments"], file=txt)

