import flask
import pickle
import json
import numpy as np
import random

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import nltk
from keras.models import load_model

app = flask.Flask(__name__, template_folder='templates')

# Open pre-trained file
model = load_model('model/chatbot_model.h5')
words = pickle.load(open('model/words.pkl', 'rb'))
classes = pickle.load(open('model/classes.pkl', 'rb'))
intents = json.loads(open('data/intents.json').read())

about_dict = json.load(open("data/resume.json"))
bot_messages = []
user_messages = []


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html', user_messages=user_messages, bot_messages=bot_messages)
    return "hello"


@app.route("/send", methods=["POST"])
def send():

    message = flask.request.form['message']
    global user_messages
    user_messages.append(message)
    response = chatbot_response(message)
    global bot_messages
    bot_messages.append(response)

    return response


if __name__ == '__main__':
    app.run()


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
