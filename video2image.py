# 将视频转换为图片

import cv2
import os

video_path1 = "dataset/videos/Video_20250627092359474.avi"
video_path2 = "dataset/videos/Video_20250627093322672.avi"
images_path1 = "dataset/videos/images1"
images_path2 = "dataset/videos/images2"

if not os.path.exists(images_path1):
    os.makedirs(images_path1)
if not os.path.exists(images_path2):
    os.makedirs(images_path2)

frame_id = 0
cap = cv2.VideoCapture(video_path1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 10 == 0:
        cv2.imwrite(f"{images_path1}/video1_Image_{frame_id}.jpg", frame)
    frame_id += 1

cap.release()

print("视频1转换为图片完成")


frame_id = 0
cap = cv2.VideoCapture(video_path2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % 10 == 0:
        cv2.imwrite(f"{images_path2}/video2_Image_{frame_id}.jpg", frame)
    frame_id += 1

cap.release()

print("视频2转换为图片完成")
