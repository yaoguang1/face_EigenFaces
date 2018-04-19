'''生成用来人脸识别的数据'''
import cv2

def generate():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    count = 0   # 计数，生成的人脸的个数
    while (True):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            '''人脸检测，裁剪灰度帧的区域，将其大小调整为200x200像素'''
            f = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            '''保存在指定的文件夹中，文件后缀名为.pgm'''
            cv2.imwrite('./data/at/yaoguang/%s.pgm' % str(count), f)
            print (count)
            count += 1

        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break
    
    camera.release()   # 释放摄像头句柄
    cv2.destroyAllWindows()   # 销毁窗口

if __name__ == "__main__":
    generate()
