from gtts import gTTS
from playsound import playsound

with open('sample.txt','r',encoding =' utf8') as f:
    text = f.read()

file_name = 'sample.mp3'
tts_ko = gTTS(text=text, lang = 'ko')
tts_ko.save(file_name)
playsound(file_name)