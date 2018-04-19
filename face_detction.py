'''用于视频中的人脸检测'''
import cv2
'''用来检测人脸的detect()函数，读取脸的特征文件，文件夹命名为cascades'''
def detect():
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./cascades/haarcascade_eye.xml')
    '''打开第一个摄像头'''
    camera = cv2.VideoCapture(0)
    while (True):
        '''捕获帧，read()会返回两个值，第一个值为布尔值，用来表明是否成功读取帧，
        第二个值为帧本身'''
        ret, frame = camera.read()
        '''捕捉到帧后，将其转换成灰度图像'''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        '''与静态图像的例子一样，将具有灰度色彩空间的帧调用detectMultiScale'''
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        '''蓝色的矩形框'''
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
            roi_gray = gray[y:y+h, x:x+w]
            '''通过限制对眼睛搜索的最小尺寸为40x40像素，可去掉所有的假阳性'''
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40,40))
            '''在人脸矩形框创建一个相对应的感兴趣区域，并在该矩形中进行“眼睛检测”'''
            '''循环输出检测眼睛的结果，并在其周围绘制绿色的矩形框'''
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
        '''在窗口中显示得到的结果'''
        cv2.imshow("camera", frame)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()
