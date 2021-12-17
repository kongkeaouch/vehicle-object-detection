from PIL import Image
import cv2
import numpy as np
import requests

image = Image.open("image.jpg")
image = image.resize((450, 250))
image_arr = np.array(image)
image
grey = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
Image.fromarray(grey)
blur = cv2.GaussianBlur(grey, (5, 5), 0)
Image.fromarray(blur)
dilated = cv2.dilate(blur, np.ones((3, 3)))
Image.fromarray(dilated)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)
Image.fromarray(closing)
car_cascade_src = 'cars.xml'
car_cascade = cv2.CascadeClassifier(car_cascade_src)
cars = car_cascade.detectMultiScale(closing, 1.1, 1)
cnt = 0
for (x, y, w, h) in cars:
    cv2.rectangle(image_arr, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cnt += 1
print(cnt, ' car detected')
Image.fromarray(image_arr)
image2 = Image.open("image2.jpg")

image2 = image2.resize((450, 250))
image_arr2 = np.array(image2)
grey2 = cv2.cvtColor(image_arr2, cv2.COLOR_BGR2GRAY)
bus_cascade_src = 'Bus_front.xml'
bus_cascade = cv2.CascadeClassifier(bus_cascade_src)
bus = bus_cascade.detectMultiScale(grey2, 1.1, 1)
cnt = 0
for (x, y, w, h) in bus:
    cv2.rectangle(image_arr2, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cnt += 1
print(cnt, ' bus detected')
Image.fromarray(image_arr2)
cascade_src = 'cars.xml'
video_src = 'Cars.mp4'
cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)
video = cv2.VideoWriter(
    'res.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (450, 250))
while True:
    ret, img = cap.read()
    if type(img) == type(None):
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)
    video.write(img)
video.release()
