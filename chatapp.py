import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from keras.models import load_model

model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

with open("resume.json") as resume:
    about_dict = json.load(resume)


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            if i['context'][0] != "":
                result += eval(i['context'][0])
            return result
    return "Something went wrong. Try again later"


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def about_education():
    msg = ""
    msg += "I went to " + about_dict['education']['university'] + " " + about_dict['education'][
        'date'] + ". \nMy degree is " + about_dict['education']['degree'] + "."
    return msg


def about_interests():
    return about_dict['interests']


def contact_info():
    email = about_dict['contact']['email']
    website = about_dict['contact']["LinkedIn"]
    contact_msg = "\nemail me - " + email + "\n find more info on LinkedIn - " + website
    return contact_msg


def list_experience():
    msg = ""
    for job in about_dict["experience"]:
        msg += "I worked at " + job["company"] + " as " + job["position"] + ".\n"
        msg += "Date: " + job['date'] + "\n"
        msg += job['about'] + "\n"
    return msg


print("welcome")
msg = input('YOU: ')
while msg != 'exit':
    print("AI: ", chatbot_response(msg))
    msg = input("YOU: ")
exit()
