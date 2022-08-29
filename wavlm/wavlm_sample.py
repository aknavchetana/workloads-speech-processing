#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import Wav2Vec2Processor, WavLMForCTC
import torch
import librosa
from IPython.display import Audio


audio1,rate1=librosa.load("h_orig.wav",sr=16000)  #audio to resample
##Audio("h_orig.wav")  # to play the audio



processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")
model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-base-plus")

inputs = processor(audio1, sampling_rate=rate1, return_tensors="pt")  #inputs to feed
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
transcription[0]
print(transcription[0])



