# 1320-blind

it literally screenshots the screen everytime and sends it to a bunch of ais; 
so open ur camera app to fullscreen to test (https://webcamtests.com/)

1. `pip install transformers google.cloud pyautogui datetime gtts playsound Image imageai torch torchvision opencv-python`
2. https://cloud.google.com/sdk/docs/install-sdk
3. https://googleapis.dev/python/google-api-core/latest/auth.html

IMPORTANT: hold ctrl c to stop the program; please stop it after few tries cuz every time use google cloud some money is used in my account (currently running on free credits given by google)

for jank2 download these: https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

(scoll down forthe 3 files; put in same folder as jank)

- kohei: detect if object left or right (and maybe if object on ground to prevent tripping)
- ngoni: detect the cane (ok objec detection cant detect idk how to detect now maybe detect th white pixels) (if possible detect what bounding boxes overlap with the cane so can say what the blind person is pointing at)
- lucy: add depth estimation to objects (https://github.com/nianticlabs/monodepth2)
- toshi: add text ocr
- ambitious: train custom object detection for the blind cane (https://manivannan-ai.medium.com/how-to-train-yolov2-to-detect-custom-objects-9010df784f36, https://manivannan-ai.medium.com/how-to-train-yolov3-to-detect-custom-objects-ccbcafeb13d2)

![image](https://cdn.discordapp.com/attachments/652418855142031361/1094896113640804393/o10042023154935.png)
