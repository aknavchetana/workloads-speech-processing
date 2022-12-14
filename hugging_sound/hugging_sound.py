# -*- coding: utf-8 -*-
"""hugging_sound.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xa0wg-rot9JLDFP4qn08FQtHPqnGDTtt
"""

#pip install huggingsound

from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["h_noise.wav"]      #input file to give

transcriptions = model.transcribe(audio_paths)

print(transcriptions)

