#!/usr/bin/env python
# coding: utf-8

# In[1]:


from transformers import Wav2Vec2Processor, HubertForCTC
from datasets import load_dataset
import torch


# In[2]:


processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForCTC.from_pretrained("facebook/hubert-large-ls960-ft")


# In[2]:


import librosa
from IPython.display import Audio


# In[3]:


audio,rate =librosa.load("YAF_youth_sad.wav",sr=16000)    #input file to include
#Audio('YAF_youth_sad.wav')


# In[7]:


input_values = processor(audio,sampling_rate=rate ,return_tensors="pt").input_values  # Batch size 1
logits = model(input_values).logits
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.decode(predicted_ids[0])


# In[8]:


print(transcription)


# In[ ]:




