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

from imageai.Detection import ObjectDetection
def localize_objects(path):
    detector = ObjectDetection()
    model = 1
    if model == 1:
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath(os.path.join(os.getcwd(), "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    elif model == 2:
        detector.setModelTypeAsYOLOv3()
        detector.setModelPath(os.path.join(os.getcwd(), "yolov3.pt"))
    elif model == 3:
        detector.setModelTypeAsTinyYOLOv3()
        detector.setModelPath(os.path.join(os.getcwd(), "tiny-yolov3.pt"))
    detector.loadModel()
    return detector.detectObjectsFromImage(input_image=path, output_image_path='O' + path)

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
        if o[i]["name"] != o[i + 1]["name"]:
            read(ONES[c] + ' ' + o[i]["name"])
            c = 1
        else:
            c = c + 1
    if len(o) != 0:
        read(ONES[c] + ' ' + o[len(o) - 1]["name"])
