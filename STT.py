import speech_recognition as sr
r = sr.Recognizer

# 마이크로부터 음성 듣기
with sr.Microphone() as source:
    print('듣고 있어요')
    audio = r.listen(source) # 마이크로부터 음성 듣기

# 파일로부터 음성 불러오기(wav, aiff/ aiff-c , flac 가능 mp3는 불가)
r = sr.Recognizer()
with sr.AudioFile('sample.wav') as source:
    print('듣고 있어요')
    audio = r.listen(source)

try:
    # 구글 API로 인식(하루 50회 제한)
    text = r.recofnize_google(audio, language = 'ko') # en-US
    print(text)
except sr.UnknownValueError:
    print('인식 실패') # 음성 인시식 실패
except sr.RequestError as e :
    print('요청 실패 : {0}'.format(e)) # api key 오류, 네트워크 단절 등


