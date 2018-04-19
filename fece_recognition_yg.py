'''加载图像数据，识别人脸'''
#coding=utf-8  
import os  
import sys  
import cv2  
import numpy as np  
  
# 图片的路径  
filepath = './faces_generate'
'''
定义2个数组，X存放每幅图片的数组列表，y存放每幅图片的序号，后面有句print函数  
可以在编译结果中看哪张图片特征最匹配实时检测到的脸，并给出置信度'''
X = []  
y = []  
  
'''读取特征图片'''
def read_images(path):  
    '''初始化计数器'''
    c = 0  
  
    '''扫描路径下的路径名，文件名，不明白的可以在下面print一下'''
    for dirname, dirnames, filenames in os.walk(path):  
        print (dirname, dirnames)
        '''提取每个文件并保存到X,y数组里'''
        for filename in filenames:  
            try:  
                '''组合路径和文件名，得到特征图的路径"./faces_generate/1.pgm"'''
                filename = os.path.join(path, filename)  
                '''把特征图以灰度图读取'''
                im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  
                
                '''重新格式化图片为200x200像素'''
                if (im is not None):  
                   im = cv2.resize(im, (200, 200))  
  
                '''把特征图片数组添加到X数组中，组成一个大的特征数组'''
                X.append(np.asarray(im, dtype=np.float32))  
                y.append(c)  
                '''输入输出错误检查'''
            except IOError:  
                print ("I/O error:") 
  
            except:  
                print ("Unexpected error:", sys.exc_info()[0])  
                raise  
            c = c + 1  
    #print X  
    #print y  
    ''''数组的维度很大'''
    return [X, y]   
  
'''
基于Eigenfaces算法的人脸识别
人脸检测开始了
'''
def face_rec():  
    '''定义一个名字的数组'''
    names = ['FaGe', 'Dazhi', 'YaoGuang']
    '''加载特征图片'''
    [x, y] = read_images(filepath)  
    '''把y数组保存为int32格式的数组，用asarry()不用开辟新的内存'''
    y = np.asarray(y, dtype=np.int32)  
    '''
    加载EigenFaceRecognizer算法，这里必须改为EigenFaceRecognizer_create
    要不会报错
    ''' 
    model = cv2.face.EigenFaceRecognizer_create()  
    '''训练数据集'''
    model.train(np.asarray(X), np.asarray(y))  
  
    '''调用人脸的特征文件'''  
    face_cascade = cv2.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')  
    camera = cv2.VideoCapture(0)  
      
    while (True):  
        read, img = camera.read()  
        faces = face_cascade.detectMultiScale(img, 1.3, 5)  
        for (x, y, w, h) in faces:  
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
            roi = gray[x: x+w, y: y+h]  
            try:  
                '''选出感兴趣的区域，使用内插法'''
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)  
                '''预测实时图片，EigenFaceRecognizer的方法'''
                params = model.predict(roi)  
                '''把匹配的特征和置信度打印在IDLE编译结果内'''
                print ("Label: %s, Confidence: %.2f" % (params[0], params[1]))  
                '''把匹配的名字显示在方框左上角，有时候会瞎显示，以后研究，还有就是现在无法显示中文字符'''
                cv2.putText(img, names[params[0]], (x, y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  
            except:  
                continue  
        cv2.imshow("camera", img)
        if cv2.waitKey(1000 // 12) & 0xff == ord("q"):  
            break  
    cv2.destroyAllWindows()  
  
if __name__ == "__main__":  
    face_rec()  