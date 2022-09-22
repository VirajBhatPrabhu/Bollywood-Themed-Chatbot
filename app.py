import json
import colorama
import keras_preprocessing.sequence
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle
import random
from colorama import Fore,Style,Back


colorama.init()

with open('intents.json') as f:
    data = json.load(f)

def chat():
    model = load_model('chatbot_model')

    with open('tokenizer.pickle','rb') as handle:
        tokenizer = pickle.load(handle)

    with open('encoder.pickle', 'rb') as enc:
        enc = pickle.load(enc)


    max_len = 20

    while True:
        print(colorama.Fore.GREEN + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == 'quit':
            break

        result = model.predict(keras_preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                                                          truncating = 'post', maxlen=max_len))

        tag = enc.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.CYAN + "ChatBot: " + Style.RESET_ALL, np.random.choice(i['responses']))



print(Fore.YELLOW + "Start talking with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
