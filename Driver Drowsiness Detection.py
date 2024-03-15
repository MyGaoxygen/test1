#!/usr/bin/env python
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import math
from cv2 import cv2
import numpy as np
from EAR import eye_aspect_ratio
from MAR import mouth_aspect_ratio
from HeadPose import getHeadTiltAndCoords

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(
    './dlib_shape_predictor/shape_predictor_68_face_landmarks.dat')

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] initializing camera...")

vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 1024
frame_height = 576

# loop over the frames from the video stream
# 2D image points. If you change the image, you need to change vector
image_points = np.array([
    (359, 391),     # Nose tip 34
    (399, 561),     # Chin 9
    (337, 297),     # Left eye left corner 37
    (513, 301),     # Right eye right corne 46
    (345, 465),     # Left Mouth corner 49
    (453, 469)      # Right mouth corner 55
], dtype="double")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# grab the indexes of the facial landmarks for the mouth
(mStart, mEnd) = (49, 68)

while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    #if frame is None:
        #break
    frame = imutils.resize(frame, width=1024, height=576)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # 检查是否检测到人脸，如果是，则绘制帧上人脸的总数。
    # 帧上的人脸数量
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 255, 0), 1)
        # 确定面部区域的面部地标，然后
        # 将面部地标 (x, y) 坐标转换为 NumPy数组
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # 提取左右眼坐标，然后使用
        # 坐标来计算双眼的长宽比
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # 将两只眼睛的长宽比加起来取平均值点击并应用
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # 检查眼球宽高比是否低于眨眼阈值
        # 阈值，如果是，则递增闪烁帧计数器
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            # 如果闭眼的次数足够多
            # 则显示警告
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "Eyes Closed!", (500, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # 否则，眼球长宽比不会低于眨眼阈值，因此重置计数器和警报
        else:
            COUNTER = 0

        mouth = shape[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
        # compute the convex hull for the mouth, then
        # 可视化嘴巴
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (650, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 如果嘴巴张开，则绘制文字
        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning!", (800, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 循环显示面部地标的 (x, y) 坐标
        # 并绘制每个地标
        for (i, (x, y)) in enumerate(shape):
            if i == 33:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[0] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 8:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[1] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 36:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[2] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 45:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[3] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 48:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[4] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            elif i == 54:
                # something to our key landmarks
                # save to our new key point list
                # i.e. keypoints = [(i,(x,y))]
                image_points[5] = np.array([x, y], dtype='double')
                # write on frame in Green
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            else:
                # everything to all other landmarks
                # write on frame in Red
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                cv2.putText(frame, str(i + 1), (x - 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        #将行列式图像点绘制到人的脸上
        for p in image_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        (head_tilt_degree, start_point, end_point, 
            end_point_alt) = getHeadTiltAndCoords(size, image_points, frame_height)

        cv2.line(frame, start_point, end_point, (255, 0, 0), 2)
        cv2.line(frame, start_point, end_point_alt, (0, 0, 255), 2)

        if head_tilt_degree:
            cv2.putText(frame, 'Head Tilt Degree: ' + str(head_tilt_degree[0]), (170, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 提取嘴部坐标，然后使用
        # 坐标来计算嘴巴的长宽比
        # 显示帧
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
