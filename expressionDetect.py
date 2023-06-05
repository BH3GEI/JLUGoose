import numpy as np
import cv2
from keras.models import load_model
from tensorflow.keras.utils import img_to_array

# 加载模型
emotion_classifier = load_model('emotionModel1.hdf5')

# 表情标签
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']


def predict_emotion(face_image):
    # 将图像转换为灰度
    gray_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # 调整图像大小以匹配模型的输入大小
    resized_image = cv2.resize(gray_image, (64, 64))

    # 将图像转换为模型所需的数组格式
    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)

    # 使用模型进行预测
    predictions = emotion_classifier.predict(image_array)

    # 返回预测结果
    return emotions[np.argmax(predictions)]


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # 使用 Haar Cascade 检测人脸
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    # 对每个检测到的人脸进行预测并绘制矩形
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face_image)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Recognition', frame)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()