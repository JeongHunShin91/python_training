import random
import json
import pickle
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
from util import *
import re
from konlpy.tag import Komoran
komoran = Komoran()

with open('intents.json', 'r', encoding = 'UTF-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []

for intent in intests['intents']:
    for pattern in intent['patterns']:
        pattern = re.sub(r'[^\w\s]','',pattern)
        w = komoran.pos(pattern)
        w = custom_morph(w)
        words.extend(w)
        documents.append((w,intent['tag']))
    if intent['tag'] not in classes:
       classes.append(intent['tag'])
words = sorted(set(words))
classes = sorted(set(classes))
pickle.dump(words, open('words.pki', 'wb'))
pickle.dump(classes, open('classes.pki','wb'))

training = []
output_empty = [0]* len(classes)
class2index = {}
for i, class_name in enumerate(classes):
    class2index[class_name]=i

for i, class_name in enumerate(classes):
    class2index[i]=class_name

for document in documents:
    bag = []
    word_patterns = document[0]

    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[class2index[document[1]]] =1
    training.append([bag,output_row])

random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])

vocab_size = len(words)
tag_size = len(classes)

model = Sequentiall()
model.add(Dense(units = 128, input_shape=(vocab_size,),activation ='relu'))
model.add(Dropout(0.5))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(tag_size, activation = 'softmax'))

sgd = SGD(lr =0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss='categorical_crossentropy', optimaizer = sgd,metrics=['acciracy'])
model.summary()

history = model.fit(train_x,train_y,epochs = 200, batch_size=5,verbose=1)
model.save('chatbot_bodel.h5', history)