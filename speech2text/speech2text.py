
import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration

import librosa

from IPython.display import Audio





model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")



# In[5]:


audio,rate =librosa.load("h_noise.wav",sr=16000)


inputs = processor(audio, sampling_rate=rate, return_tensors="pt")
generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

transcription = processor.batch_decode(generated_ids)
print(transcription)
