import cv2
import numpy as np

img1=cv2.imread('ImagesQuery/Three Body.jpg',0)
img2=cv2.imread('ImagesTrain/T.jpg',0)
img1=cv2.resize(img1,(int(img1.shape[1]/3),int(img1.shape[0]/3)))
img2=cv2.resize(img2,(int(img2.shape[1]/5),int(img2.shape[0]/5)))

#创建特征检测器
orb=cv2.ORB_create(nfeatures=1000)

#调用特征检测器寻找特征
kp1,des1=orb.detectAndCompute(img1,None)#None代表不添加遮罩，kp代表特征点，des代表描述符（他的shape(m，n)的含义表示有m个特征，每个特征由n个值描述）
kp2,dest2=orb.detectAndCompute(img2,None)

#绘制特征点
imgKp1=cv2.drawKeypoints(img1,kp1,None)
imgKp2=cv2.drawKeypoints(img2,kp2,None)

#生成描述符匹配器，开始匹配，获取匹配结果
bf=cv2.BFMatcher()
matches=bf.knnMatch(des1,dest2,k=2)#k等于2表示返回两个值供我们进行筛选

#遍历匹配结果，获取最佳的匹配点对
good=[]
for m,n in matches:#m,n即返回的两个值
    if m.distance<0.75*n.distance:
        good.append([m])

print(len(good))

img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)




# cv2.imshow('Kp1',imgKp1)
# cv2.imshow('Kp2',imgKp2)
# cv2.imshow('img1',img1)
# cv2.imshow('img2',img2)
cv2.imshow('img3',img3)
cv2.waitKey(0)