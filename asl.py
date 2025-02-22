import streamlit as st
import pandas as pd
import numpy as np
import cv2

st.title('ASL Translator')

st.camera_input("sign", label_visibility="hidden")

vid=cv2.VideoCapture(0)
if not vid.isOpened():
    print('cannot open the camera')
    exit()
while True:
    ret,cap=vid.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    hand=hand.detectMultiScale(gray, 1.05, 5)
    if hand is ():
        print('no hand detected')

    # for (x,y,a,b) in hand:
    #     cv2.rectangle(cap,(x,y),(x+a,y+b),(127,0,255),2)
    #     roi_color=cap[y:y+b,x:x+a]
    #     img = cv2.resize(roi_color, (220,220), interpolation=cv2.INTER_CUBIC)
    #     img = img.astype('float16')
    #     img = np.expand_dims(img, axis=0)
    #     score=predict(model,img)
    #     if score==1:
    #         cv2.putText(cap,'Male',(x+a,y+b), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,0,0),1, cv2.LINE_AA)
    #     if score==0:
    #         cv2.putText(cap,'Female',(x+a,y+b), cv2.FONT_HERSHEY_SIMPLEX ,0.5,(255,0,0),1, cv2.LINE_AA)
    #     cv2.imshow('image',cap)
    #     cv2.waitKey(2)
    if cv2.waitKey(2) & 0xFF==ord('q'):
        break
vid.release()
cv2.destroyAllWindows()