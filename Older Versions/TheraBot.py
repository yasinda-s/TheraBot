#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import itertools, pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes = ["neutral", "happy", "sad", "love", "anger"]


# In[2]:


model = load_model("therabot.h5")


# In[9]:


emotion_scores = {
    "neutral" : 0,
    "happy" : 0,
    "sad" : 0,
    "love" : 0,
    "anger" : 0
}

def emotion_score(y_prob):
    emotion_scores["neutral"] += y_prob[0][0]
    emotion_scores["happy"] += y_prob[0][1]
    emotion_scores["sad"] += y_prob[0][2]
    emotion_scores["love"] += y_prob[0][3]
    emotion_scores["anger"] += y_prob[0][4]
    
def get_highest_key():
  highest_value = 0
  for key in emotion_scores.keys():
    if emotion_scores[key] >= highest_value:
      highest_value = emotion_scores[key]
      highest_key = key
  return highest_key

def consolidation_message(highest_key):
    if highest_key == 'neutral':
      print("Therabot: You are Neutral")
    elif highest_key == 'happy':
      print("Therabot: You are Happy!")
    elif highest_key == 'sad':
      print("Therabot: You are Sad")
    elif highest_key == 'love':
      print("Therabot: You are Love")
    elif highest_key == 'anger':
        print("Therabot: You are Anger")
    else:
      print("Goodbye! Take care")
    
def reply(detected_intent):
    if detected_intent == 'neutral':
      print("Therabot: Neutral detected")
    elif detected_intent == 'happy':
      print("Therabot: Happy detected!")
    elif detected_intent == 'sad':
      print("Therabot: Sad detected")
    elif detected_intent == 'love':
      print("Therabot: Love detected")
    elif detected_intent == 'anger':
        print("Therabot: Anger detected")
    
def fallback_intent():
    print("Sorry I don't understand. Can you elaborate please?")
    
def analyze_message(user_message):
    text = [user_message]
    sequences_test = tokenizer.texts_to_sequences(text)
    MAX_SEQUENCE_LENGTH = 30
    data_int_t = pad_sequences(sequences_test, padding='pre', maxlen=(MAX_SEQUENCE_LENGTH-5))
    data_test = pad_sequences(data_int_t, padding='post', maxlen=(MAX_SEQUENCE_LENGTH))
    return data_test

def predict_emotion(data_test, y_prob):
#     y_prob = model.predict(data_test)
    for n, prediction in enumerate(y_prob):
        pred = y_prob.argmax(axis=-1)[n]
#         print(y_prob[0])
    return pred


# In[10]:


from keras.preprocessing.sequence import pad_sequences

print("Therabot: Hi, my name is TheraBot")
print("Therabot: What is your name?")
user_name = input()
while True:
    print("User: ", end="")
    user_message = input()
    if user_message!="quit":
        data_test = analyze_message(user_message)
        y_prob = model.predict(data_test)
        pred = predict_emotion(data_test,y_prob)
        #print(y_prob[0])
        highest_emotion_confidence = y_prob[0][pred]
        if highest_emotion_confidence > 0.33:
            reply(classes[pred])
        else:
            fallback_intent()
        emotion_score(y_prob)
    elif user_message.lower() == "quit":
        highest_key = get_highest_key()
        consolidation_message(highest_key)
        break


# In[ ]:




