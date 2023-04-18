import sounddevice as sd
import soundfile as sf
from gtts import gTTS 
from datetime import datetime
import os
from transformers import pipeline
import pyautogui
from PIL import Image
from PIL import ImageDraw
import speech_recognition as sr
import io
from scipy.io.wavfile import write
from playsound import playsound
import wavio as wv

def read(s):
    audio = gTTS(text = s, lang = "en", slow = False)
    audiopath = 'tmp/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.mp3'
    audio.save(audiopath)
    playsound(audiopath)
    #os.remove(name)

duration = 10
sd.default.samplerate = 44100
sd.default.channels = 1
#sd.default.device = 'Speakers (Realtek(R) Audio), MME'

read('starting up')

image_to_text = pipeline(task = "image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
def gen_desc(image):
    return image_to_text(image)[0]["generated_text"]

object_detector = pipeline(model = "facebook/detr-resnet-50")
def detect_object(image):
    objs = object_detector(image)
    print(objs)

image_segmenter = pipeline(model = "nvidia/segformer-b0-finetuned-ade-512-512")
def segment_image(image):
    segs = image_segmenter(image)
    print(segs)

#image_segmenter_city = pipeline(model = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
#def segment_image_city(image):
#    segs = image_segmenter_city(image)
#    print(segs)

depth_estimator = pipeline(task = "depth-estimation", model = "Intel/dpt-large")
def estimate_depth(image):
    deps = depth_estimator(image)
    print(deps)

recognizer = sr.Recognizer()
def recognize(audio_array):
    audiopath = 'tmp/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.wav'
    write(audiopath, 44100, audio_array)
    audiofile = sr.AudioFile(audiopath)
    with audiofile as source:
        audiodata = recognizer.record(source)
        try:
            return recognizer.recognize_google(audiodata)
        except sr.UnknownValueError:
            return ""

def process(image, command):
    print(command)
    detect_object(image)
    segment_image(image)
    #segment_image_city(image)
    estimate_depth(image)

read("start up complete")

test = False;
if test:
    for f in os.listdir('test/'):
        with image.open('test/' + f) as im:
            process(im, "")
else:
    command = ''
    while True:
        recording = sd.rec(int(duration * sd.default.samplerate), channels = 1, dtype = 'int32')
        ss = pyautogui.screenshot()
        sspath = 'out/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png'
        ss.save(sspath)
        process(ss, command)
        sd.wait()
        command = recognize(recording)
