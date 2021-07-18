import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tkinter import *
from tensorflow.keras.models import load_model
from time import gmtime, strftime

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pk1', 'rb'))
classes = pickle.load(open('classes.pk1', 'rb'))
model = load_model('chatbotmodel.h5')
chat_bot = "BalanceBot"

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list =[]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

EMPTY_SPACE_COLOR = "#DEEFF5"
TEXT_COLOR = "#17202A"
LABEL_BG_COLOR = "Black"
CONTRAST_COLOR = "#90EE90"
DATETIME_COLOR = '#D3D3D3'

FONT = "Comic 13 bold"
FONT_DATETIME = "Comic 12 bold"

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat")
        self.window.resizable(width=False, height=False)
        self.window.configure(width=700, height=550, bg=EMPTY_SPACE_COLOR)

        # head label
        head_label = Label(self.window, bg=LABEL_BG_COLOR, fg=EMPTY_SPACE_COLOR,
                        text="BalanceBot", font=FONT, pady=10)
        head_label.place(relwidth=1)

        # tiny divider
        line = Label(self.window, width=450, bg=CONTRAST_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        # text widget
        self.text_widget = Text(self.window, width=20, height=2, bg=EMPTY_SPACE_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=15, pady=5, wrap="word")
        self.text_widget.place(relheight=0.81, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor="arrow", state=DISABLED)     

        # scroll bar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.99)
        scrollbar.configure(command=self.text_widget.yview)
        
        # bottom label
        bottom_label = Label(self.window, bg=LABEL_BG_COLOR, height=50)
        bottom_label.place(relwidth=1, rely=0.885)

        # Message Entry
        self.msg_entry = Entry(bottom_label, bg=EMPTY_SPACE_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        msg1 = "Available Commands: Motivational Tips, Quote, Fact\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # Send button
        send_button = Button(bottom_label, text="Send", fg=LABEL_BG_COLOR, font=FONT, width=20, bg=CONTRAST_COLOR,
                             command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")

    def _insert_message(self, msg, sender):
        if not msg:
            return

        date_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        self.msg_entry.delete(0, END)
        msgDateTime1 = f"{date_time}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msgDateTime1)
        self.text_widget.configure(state=DISABLED)
  
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)

        msgDateTime2 = f"{date_time}\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msgDateTime2)
        self.text_widget.configure(state=DISABLED)
        
        msg2 = f"{chat_bot}: {get_response(predict_class(msg), intents)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
