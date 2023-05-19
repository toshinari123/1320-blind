import sounddevice as sd
import soundfile as sf
from gtts import gTTS 
from datetime import datetime
import os
from transformers import pipeline
import pyautogui
from PIL import Image, ImageOps, ImageDraw, ImageFont
import speech_recognition as sr
import io
from scipy.io.wavfile import write
import numpy as np
import time
import psutil

def read(s):
    audio = gTTS(text = s, lang = "en", slow = False)
    audiopath = 'tmp/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.wav'
    audio.save(audiopath)
    data, fs = sf.read(audiopath)
    #sd.default.device = 'CABLE Input (VB-Audio Virtual C, MME'
    #sd.default.device = 'Speakers (Realtek(R) Audio), MME'
    sd.default.device = 'Headphones (XO-F21 Stereo), MME'
    sd.play(data, fs)
    #os.remove(name)

duration = 5
sd.default.samplerate = 44100
sd.default.channels = 1

read('starting up')

text_to_read = ""
image_to_text = pipeline(task = "image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
def gen_desc(image):
    return image_to_text(image)[0]["generated_text"]

object_detector = pipeline(model = "facebook/detr-resnet-50")
font = ImageFont.truetype("consola.ttf", size = 30)
def detect_object(image):
    global text_to_read
    objs = object_detector(image)
    image = image.copy()
    print(objs)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for obj in objs:
        rect = [(obj['box']['xmin'], obj['box']['ymin']), (obj['box']['xmin'], obj['box']['ymax']), (obj['box']['xmax'], obj['box']['ymax']), (obj['box']['xmax'], obj['box']['ymin'])]
        draw.polygon(rect)
        draw.text(rect[0], obj['label'] + " " + str(obj['score']), font = font)
    labels = [obj['label'] for obj in objs]
    for label in set(labels):
        label_objs = [obj for obj in objs if obj['label'] == label]
        text_to_read += english_number[len(label_objs)] + ' ' + label + ':'
        for x in label_objs:
            if (x['box']['xmax'] - x['box']['xmin']) * (x['box']['ymax'] - x['box']['ymin']) > 40000:
                text_to_read += 'one ' + label
                if x['box']['xmax'] < width / 2:
                    text_to_read += ' on the left '
                elif x['box']['xmin'] > width / 2:
                    text_to_read += ' on the right   '
                else:
                    text_to_read += ' in the middle   '
    image.save('out/detect_object' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png')


image_segmenter = pipeline(model = "nvidia/segformer-b0-finetuned-ade-512-512")
col = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), 
(255, 64, 64), (64, 255, 64), (64, 64, 255), (255, 255, 64), (255, 64, 255), (64, 255, 255), 
(128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), 
(255, 200, 200), (200, 255, 200), (200, 200, 255), (255, 255, 200), (255, 200, 255), (200, 255, 255), 
(255, 128, 128), (128, 255, 128), (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255), ]
ground = ['floor', 'sidewalk', 'stairs', 'stairway', 'earth']
depth_estimator = pipeline(task = "depth-estimation", model = "Intel/dpt-large")
english_number = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
def segment_and_estimate_depth(image):
    global text_to_read
    segs = image_segmenter(image)
    print(segs)
    visual = image
    width, height = image.size
    left = False
    right = False
    straight = False
    
    for (ind, seg) in enumerate(segs):
        arr = np.array(seg['mask'].getdata())
        if np.count_nonzero(arr) > 0.8 * arr.shape[0]:
            text_to_read += " warning: " + seg['label'] + ' more than eighty percent of vision   '
        visual = Image.composite(visual, Image.new("RGB", image.size, col[ind]), ImageOps.invert(seg["mask"]))
        if seg["mask"].getpixel((0, height - 1)) != 0 and seg["label"] in ground:
            left = False
        if seg["mask"].getpixel((width - 1, height - 1)) != 0 and seg["label"] in ground:
            right = False
        if seg["mask"].getpixel((int(width / 2), int(height / 3 * 2))) != 0 and seg["label"] in ground:
            straight = False
    
    if left:
        text_to_read += " navigation: can go left."
    if right:
        text_to_read += " navigation: can go right."
    if straight:
        text_to_read += " navigation: can go straight "
    if not left and not right and not straight:
        text_to_read += " navigation: blocked "
    
    draw = ImageDraw.Draw(visual)
    for (ind, seg) in enumerate(segs):
        draw.rectangle([10, ind * 30, 200, (ind + 1) * 30], fill = (255, 255, 255))
        draw.text((10, ind * 30), seg["label"], fill = col[ind], font = font)
    visual.save('out/visualization' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png')
    
    deps = depth_estimator(image)
    print(deps)
    deps['depth'] = deps['depth'].convert('RGB')
    width, height = image.size
    draw = ImageDraw.Draw(deps['depth'])
    print(deps['predicted_depth'].size())
    r = 0.0
    for i in range(0, 384):
        r += (1 / max(0.000001, float(deps['predicted_depth'][0][383][i]))) / 384
    r = 2 / r
    for i in range(0, 384, 30):
        for j in range(0, 384, 30):
            draw.text((int(width / 384 * j), (int(height / 384 * i))), str(round(1 / max(0.000001, float(deps['predicted_depth'][0][i][j])) * r, 3)), fill = (255, 0, 0), font = font)
    deps['depth'].save('out/depth' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png')

    d = {'ground': 0, 'door': 0, 'stairs': 0, 'stairway': 0}
    left = []
    right = []
    for seg in segs:
        d[seg['label']] = 0
    for i in range(0, 384, 5):
        for j in range(0, 384, 5):
            for seg in segs:
                if seg["mask"].getpixel((int(width / 384 * j), (int(height / 384 * i)))) != 0 and float(deps['predicted_depth'][0][i][j]) != 0:
                    if seg["label"] in ground:
                        d['ground'] = max(d['ground'], round(1 / float(deps['predicted_depth'][0][i][j])))
                    d[seg['label']] = max(d[seg['label']], round(1 / float(deps['predicted_depth'][0][i][j])))
                    if j < 384 / 2:
                        left += [seg['label']]
                    if j > 384 / 2:
                        right += [seg['label']]
    if d['ground'] != 0:
        text_to_read += ' furthest ground is ' + english_number[d['ground']] + 'meters away   '
    if d['door'] != 0:
        text_to_read += ' door' + english_number[d['ground']] + 'meters away '
        if 'door' in left:
            if 'door' in right:
                text_to_read += 'straight ahead, '
            else:
                text_to_read += 'to the left, '
        else:
            if 'door' in right:
                text_to_read += 'to the right, '
    if d['stairs'] != 0 or d['stairway'] != 0:
        text_to_read += ' stairs' + english_number[max(d['stairs'], d['stairway'])] + 'meters away '   
        if 'stairs' in left or 'stairway' in left:
            if 'stairs' in right or 'stairway' in left:
                text_to_read += 'straight ahead, '
            else:
                text_to_read += 'to the left, '
        else:
            if 'stairs' in right or 'stairway' in left:
                text_to_read += 'to the right, '

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

from google.cloud import vision
client = vision.ImageAnnotatorClient()
text_detected = ''
def google(image):
    width, height = image.size
    imagepath = 'out/crop' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png'
    image.crop((width / 10, height / 10, width / 10 * 9, height / 10 * 9)).save(imagepath)
    global text_to_read
    global text_detected
    with open(imagepath, 'rb') as image_file:
        content = image_file.read()
    image2 = vision.Image(content=content)
    response = client.text_detection(image=image2)
    texts = response.text_annotations
    if len(texts) != 0:
        text_detected = texts[0].description.replace(r'\n', ' newline ')
        print(texts[0])
        if len(text_detected) > 60:
            text_to_read += ' text too long: say voice command read to read text'
        else:
            text_to_read += ' text in vision: ' + text_detected + ' text ended'
    response = client.logo_detection(image=image2)
    logos = response.logo_annotations
    print(logos)


on = True;
t = time.time()
def process(image, command):
    global on
    global text_to_read
    global t
    global text_detected
    print('COMMAND')
    command = command + '#'
    print(command)
    if 'start' in command:
        read('Starting')
        sd.wait()
        on = True
    if 'stop' in command:
        read('Stopping')
        sd.wait()
        on = False
    if not on:
        return
    if 'help' in command:
        text_to_read += ' available voice commands: help, start, stop, remind, reminder, describe, picture, read   '
    if 'describe' in command:
        text_to_read += ' description: you are seeing ' + gen_desc(image) + '   '
    if 'picture' in command:
        text_to_read += ' saving the picture   '
        image.save('pics/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png')
    if 'read' in command:
        text_to_read += ' text in vision: ' + text_detected + ' text ended'
    if 'record' in command:
        read('recording 30 seconds:')
        sd.wait()
        sd.default.device = 'Line 1 (Virtual Audio Cable), MME'
        #sd.default.device = 'Microphone Array (Realtek(R) Au'
        recording = sd.rec(int(30 * sd.default.samplerate), channels = 1, dtype = 'int32')
        sd.wait()
        text_to_read = 'recording ended'
        sd.wait()
        write('rem/reminder.wav', 44100, recording)
    if 'play' in command:
        data, fs = sf.read('rem/reminder.wav')
        #sd.default.device = 'CABLE Input (VB-Audio Virtual C, MME'
        sd.default.device = 'Speakers (Realtek(R) Audio), MME'
        sd.play(data, fs)
        sd.wait()
        text_to_read = 'recording ended'
    if text_to_read == '':
        text_to_read = 'no command detected'
    if on:
        read(text_to_read)
    sd.wait()
    text_to_read = ''

    print('other processes: ' + str(time.time() - t) + 'seconds')
    t = time.time()
    detect_object(image)
    print('detect object: ' + str(time.time() - t) + 'seconds')
    t = time.time()
    #segment_and_estimate_depth(image)
    #print('segment and depth: ' + str(time.time() - t) + 'seconds')
    t = time.time()
    google(image)
    print('google: ' + str(time.time() - t) + 'seconds')
    t = time.time()

read("start up complete")

test = False;
if test:
    for f in os.listdir('test/'):
        with Image.open('test/' + f) as im:
            process(im, "")
else:
    command = ''
    while True:
        read('say voice command')
        sd.wait()
        sd.default.device = 'Line 1 (Virtual Audio Cable), MME'
        #sd.default.device = 'Microphone Array (Realtek(R) Au'
        recording = sd.rec(int(duration * sd.default.samplerate), channels = 1, dtype = 'int32')
        sd.wait()
        write('tmp/command' + datetime.now().strftime("%d%m%Y%H%M%S") + '.wav', 44100, recording)
        command = recognize(recording)
        text_to_read = ''
        ss = pyautogui.screenshot()
        sspath = 'out/' + datetime.now().strftime("%d%m%Y%H%M%S") + '.png'
        ss.save(sspath)
        process(ss, command)
        if text_to_read == '':
            text_to_read = 'no object detected'
        read(text_to_read)
        sd.wait()
