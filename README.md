# 1320-blind

it literally screenshots the screen everytime and sends it to a bunch of ais; 
so open ur camera app to fullscreen to test

1. `pip install transformers google.cloud pyautogui datetime gtts playsound`
2. https://cloud.google.com/sdk/docs/install-sdk
3. https://googleapis.dev/python/google-api-core/latest/auth.html

- kohei: detect if object left or right (and maybe if object on ground to prevent tripping)
- ngoni: detect the cane (if possible detect what bounding boxes overlap with the cane so can say what the blind person is pointing at)
- lucy: add depth estimation to objects (https://github.com/nianticlabs/monodepth2)
- toshi: add text ocr

