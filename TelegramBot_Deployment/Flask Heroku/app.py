#!/usr/bin/env python
# coding: utf-8

from flask import Flask, request
import telegram
from credentials import botToken, botUserName, URL

import numpy as np
np.random.seed(42)
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from cv2 import imdecode, resize, cvtColor, COLOR_BGR2GRAY
import io, requests

global bot
global TOKEN
TOKEN = botToken
bot = telegram.Bot(token=TOKEN)

modelEmotion = load_model('cnnModelEmotion.h5')
modelGender = load_model('cnnModelGender.h5')
modelAge = load_model('cnnModelAge.h5')

app = Flask(__name__)

@app.route('/{}'.format(TOKEN), methods=['POST'])
def start():
    # retrieve the message in JSON and then transform it to Telegram object
    update = telegram.Update.de_json(request.get_json(force=True), bot)
    chat_id = update.message.chat.id
    msg_id = update.message.message_id
    imageId = update.message.photo[len(update.message.photo)-1].file_id
    bot.sendMessage(chat_id=chat_id, text='Analysing image, please wait')

    url = bot.get_file(imageId).file_path
    imgStream = io.BytesIO(requests.get(url).content)
    photo = imdecode(np.fromstring(imgStream.read(), np.uint8), 1)

    detected = False
    results = MTCNN().detect_faces(photo)
    if len(results)!=0:
        detected=True
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = photo[y1:y2, x1:x2]
        face = resize(face,(150,150))
        face = cvtColor(face,COLOR_BGR2GRAY)
        x = (face-face.mean())/face.std()
        x = np.array(x,dtype=np.float32)
        x = x.reshape(1,150,150,1)
        resultEmotion = (modelEmotion.predict(x)).tolist()[0]
        emotionsDict = {0:'Happy', 1:'Neutral', 2:'Sad/Angry', 3:'Surprised/Fearful'}
        emotion = emotionsDict[resultEmotion.index(max(resultEmotion))]
        resultGender = int(round(modelGender.predict(x).tolist()[0][0]))
        genderDict = {0:'Female', 1:'Male'}
        gender = genderDict[resultGender]
        age = int(round(modelAge.predict(x).tolist()[0][0]))
        answer = 'Emotion: ' + str(emotion) + ', Gender: ' + str(gender) + ', Age: ' + str(age)
        bot.sendMessage(chat_id=chat_id, text=str(answer))
    else:
        bot.sendMessage(chat_id=chat_id, text='Sorry, no face detected')

    return 'ok'

@app.route('/setwebhook', methods=['GET', 'POST'])
def set_webhook():
    s = bot.setWebhook('{URL}{HOOK}'.format(URL=URL, HOOK=TOKEN))
    if s:
        return "webhook setup ok"
    else:
        return "webhook setup failed"

@app.route('/')
def index():
    return '.'

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0')

