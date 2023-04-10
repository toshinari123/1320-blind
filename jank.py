import sys
sys.path.append('C:\\users\\toshi\\appdata\\local\\packages\\pythonsoftwarefoundation.python.3.9_qbz5n2kfra8p0\\localcache\\local-packages\\python39\\site-packages')

from gtts import gTTS 
from playsound import playsound
from datetime import datetime
import os
def read(s):
    audio = gTTS(text = s, lang = "en", slow = False)
    name = datetime.now().strftime("%d%m%Y%H%M%S") + '.mp3'
    audio.save(name)
    playsound(name)
    os.remove(name)

read("starting up")

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
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
    return objects

read("start up complete")

import pyautogui
from PIL import Image
from PIL import ImageDraw
while True:
    ss = pyautogui.screenshot()
    name = datetime.now().strftime("%d%m%Y%H%M%S") + '.png'
    ss.save(name)
    desc = gen_desc(name)
    print(desc)
    read(desc)
    o = localize_objects(name)
    c = 1
    ONES = ["zero","one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen","eighteen","nineteen"]
    for i in range(len(o) - 1):
        if o[i].name != o[i + 1].name:
            read(ONES[c] + ' ' + o[i].name)
            c = 1
        else:
            c = c + 1
    if len(o) != 0:
        read(ONES[c] + ' ' + o[len(o) - 1].name)
    img = Image.open(name).convert('RGBA')
    img2 = ImageDraw.Draw(img)
    width, height = img.size
    for obj in o:
        rect = list(map(lambda v : (int(v.x * width), int(v.y * height)), obj.bounding_poly.normalized_vertices))
        print(rect)
        img2.polygon(rect)
        img2.text(rect[0], obj.name + " " + str(obj.score))
    img.save('o' + name)
