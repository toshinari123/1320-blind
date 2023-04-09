import sys
sys.path.append('C:\\users\\toshi\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\\localcache\\local-packages\\python39\\site-packages')

from transformers import pipeline
image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
def gen_desc(image):
    return image_to_text(image)[0]["generated_text"]

from google.cloud import vision
def localize_objects(path):
    client = vision.ImageAnnotatorClient()
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    objects = client.object_localization(
        image=image).localized_object_annotations
    names = []
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        names = names + [object_.name]
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
    print(names)      
    return names

from gtts import gTTS 
from playsound import playsound
def read(s):
    audio = gTTS(text = s, lang = "en", slow = False)
    name = datetime.now().strftime("%d%m%Y%H%M%S") + '.mp3'
    audio.save(name)
    #AudioSegment.from_mp3(name)
    #uh delete this audio file
    playsound(name)

import pyautogui
from datetime import datetime
while True:
    ss = pyautogui.screenshot()
    name = datetime.now().strftime("%d%m%Y%H%M%S") + '.png'
    ss.save(name)
    read(gen_desc(name))
    o = localize_objects(name)
    o = o + ['######']
    c = 1
    ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
    for i in range(len(o) - 1):
        if o[i] != o[i + 1]:
            read(ONES[c] + ' ' + o[i])
            c = 1
        else:
            c = c + 1
