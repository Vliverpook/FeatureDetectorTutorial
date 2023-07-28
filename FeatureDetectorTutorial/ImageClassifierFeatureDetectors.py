#此py为通用方法，即可以批量处理所有的图像
import cv2
import os
import numpy as np

path='ImagesQuery'
orb=cv2.ORB_create(nfeatures=1000)
images=[]
classNames=[]

#获取路径下的图像文件名
myList=os.listdir(path)
print('Total Classes Detected',len(myList))

#导入图像
for cl in myList:
    imgCur=cv2.imread(f'{path}/{cl}',0)
    images.append(imgCur)
    #去除后缀名
    classNames.append(os.path.splitext(cl)[0])

print(classNames)

#获取描述符
def findDes(images):
    desList=[]
    for img in images:
        kp,des=orb.detectAndCompute(img,None)
        desList.append(des)

    return desList

#根据描述符进行匹配，返回最佳匹配下标
def findID(img,desLis,thres=15):
    kp2,des2=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher()
    findVal=-1
    matchList=[]
    try:
        for des in desLis:
            matches=bf.knnMatch(des,des2,k=2)#k等于2表示返回两个值供我们进行筛选
            good = []
            for m, n in matches:  # m,n即返回的两个值
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass

    if len(matchList)!=0:
        if max(matchList) > thres:
            findVal=matchList.index(max(matchList))

    return findVal

desList=findDes(images)

print(len(desList))

cap = cv2.VideoCapture(0)

while True:

    success, img=cap.read()
    imgOriginal=img.copy()
    img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    id=findID(img2,desList)
    if id!=-1:
        cv2.putText(imgOriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

    cv2.imshow('img1',imgOriginal)
    cv2.waitKey(1)

# #创建特征检测器
# orb=cv2.ORB_create(nfeatures=1000)
#
# #调用特征检测器寻找特征
# kp1,des1=orb.detectAndCompute(img1,None)#None代表不添加遮罩，kp代表特征点，des代表描述符（他的shape(m，n)的含义表示有m个特征，每个特征由n个值描述）
# kp2,dest2=orb.detectAndCompute(img2,None)
#
# #绘制特征点
# imgKp1=cv2.drawKeypoints(img1,kp1,None)
# imgKp2=cv2.drawKeypoints(img2,kp2,None)
#
# #生成描述符匹配器，开始匹配，获取匹配结果
# bf=cv2.BFMatcher()
# matches=bf.knnMatch(des1,dest2,k=2)#k等于2表示返回两个值供我们进行筛选
#
# #遍历匹配结果，获取最佳的匹配点对
# good=[]
# for m,n in matches:#m,n即返回的两个值
#     if m.distance<0.75*n.distance:
#         good.append([m])

# print(len(good))
#
# img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
#
#
#
#
# # cv2.imshow('Kp1',imgKp1)
# # cv2.imshow('Kp2',imgKp2)
# # cv2.imshow('img1',img1)
# # cv2.imshow('img2',img2)
# cv2.imshow('img3',img3)
# cv2.waitKey(0)