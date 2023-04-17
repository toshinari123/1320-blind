# 1320-blind

In our project, we demonstrate a prototypical smart glasses to help blind people navigate and identify object easier.

components:
- smart glasses: streams video and audio to the internet by connecting to a wifi network. To enable transportability one can connect it to a mobile hotspot instead.
- backend / server / computer: receives the video and audio streams; analyze the audio stream for voice commands and sends the video data to a bunch of ai to convert to useful text.
- in the prototype: the text is read aloud by the computer and the output audio is streamed to the blind persons phone throuugh the internet
- ideally: the text is sent to a custom app on the blind persons phone and it reads out the texts

note on image / video:
- in the prototype: an image is taken from the video stream every 10 seconds and the image is sent to different single-image processing machine learning models
- ideally: the video stream itself is taken as the input to more sophisticated machine learning models in order to take advantage of the temporal relations between frames

voice commands:
- [nothing]: every 10 seconds it reads aloud what object it detected and if the objects are to the left, in front, or to the right.
- "describe": sends the image to a natural languge processing model to generate a coherent description of the image.
- "picture": saves the image and in the prototype saves it to a google drive for future use of the blind person; ideally the image will be sent to the custom app
- "crop" (todo): crops image to a rectangular object such as a poster or a paper the user is holding; describes the cropped image and saves it as well
- "point" (todo): reads aloud what object the user is pointing at with the blind cane

# instructions

it literally screenshots the screen everytime and sends it to a bunch of ais; 
so open ur camera app to fullscreen to test (https://webcamtests.com/)

install libraries: 

1. `pip install sounddevice soundfile gtts transformers pyautogui pillow speechrecognition scipy torch`

for gcloud:

1. https://cloud.google.com/sdk/docs/install-sdk
2. https://googleapis.dev/python/google-api-core/latest/auth.html

for imageai:

1. download the 3 .pt files on this webpage (scroll down a bit) https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection
2. put in same folder as the python scripts

for huggingface:

1. no setup required we will use this as the main one

IMPORTANT: hold ctrl c to stop the program; please stop it after few tries cuz every time use google cloud some money is used in my account (currently running on free credits given by google)

job division:

- kohei: detect if object left or right (and maybe if object on ground to prevent tripping)
- ngoni: detect the cane (ok objec detection cant detect idk how to detect now maybe detect th white pixels) (if possible detect what bounding boxes overlap with the cane so can say what the blind person is pointing at)
- lucy: add depth estimation to objects (https://github.com/nianticlabs/monodepth2)
- toshi: add text ocr
- ambitious: train custom object detection for the blind cane (https://manivannan-ai.medium.com/how-to-train-yolov2-to-detect-custom-objects-9010df784f36, https://manivannan-ai.medium.com/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2)

![image](https://cdn.discordapp.com/attachments/652418855142031361/1094896113640804393/o10042023154935.png)

# credits

- https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
- google cloud vision
- https://github.com/OlafenwaMoses/ImageAI
- https://arxiv.org/abs/1708.02002
- https://arxiv.org/abs/1804.02767
- https://github.com/NVlabs/SegFormer
- https://arxiv.org/abs/2105.15203
- https://github.com/nianticlabs/monodepth2
