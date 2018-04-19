'''查看生成的人脸特征图'''
import cv2  

'''打开C盘路径下保存图片的22.pgm文件，并保存为灰度图像'''
img = cv2.imread('C:\\Users\\yao\\Desktop\\faces_generate\\22.pgm', cv2.IMREAD_GRAYSCALE)  
'''
顺便看看图片的格式，好大的一个列表对象，  
里面的数组代表了图片上一个个行和列上的像素，格式是[xxx,xxx,xxx]  
xxx = 0~255'''  

print(img)
# 在名为img的窗口上显示图片，像素为200x200  
print(img.shape)

cv2.imshow('img',img)  
cv2.waitKey()  
cv2.destroyAllWindows()