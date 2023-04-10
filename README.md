# 1320-blind

it literally screenshots the screen everytime and sends it to a bunch of ais; 
so open ur camera app to fullscreen to test

1. `pip install transformers google.cloud pyautogui datetime gtts playsound Image`
2. https://cloud.google.com/sdk/docs/install-sdk
3. https://googleapis.dev/python/google-api-core/latest/auth.html

IMPORTANT: hold ctrl c to stop the program; please stop it after few tries cuz every time use google cloud some money is used in my account (currently running on free credits given by google)

- kohei: detect if object left or right (and maybe if object on ground to prevent tripping)
- ngoni: detect the cane (if possible detect what bounding boxes overlap with the cane so can say what the blind person is pointing at)
- lucy: add depth estimation to objects (https://github.com/nianticlabs/monodepth2)
- toshi: add text ocr

![image](https://cdn.discordapp.com/attachments/652418855142031361/1094896113640804393/o10042023154935.png)
