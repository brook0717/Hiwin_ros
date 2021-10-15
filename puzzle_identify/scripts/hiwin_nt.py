#!/usr/bin/env python3
# license removed for brevity
import sys
#sys.path.remove("/opt/ros/melodic/lib/python2.7/dist-packages/")
#sys.path.insert(1, "/usr/local/lib/python3.6/dist-packages/")
sys.path.insert(1, "/home/user/.local/lib/python3.6/site-packages/")
sys.path.insert(0, '/opt/installer/open_cv/cv_bridge/lib/python3/dist-packages/')
import pyrealsense2 as rs
import cv2 
import threading
import time
import rospy
import os
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
#from ROS_Socket.srv import *
#from ROS_Socket.msg import *
import math
import enum
#import Hiwin_RT605_Socket as ArmTask
from std_msgs.msg import Int32MultiArray    
from matplotlib import pyplot as plt
from vision.msg import haha


def talker():
    # pub = rospy.Publisher('chatter', haha, queue_size=10)
    # rospy.init_node('hiwin_nt', anonymous=True)
    # pub = rospy.Publisher('apple', haha, queue_size=10)
    
    value = haha()
    value.cnt_ux1 = list_cnt_ux[0]
    value.cnt_ux2 = list_cnt_ux[1]
    value.cnt_ux3 = list_cnt_ux[2]
    value.cnt_ux4 = list_cnt_ux[3]
    value.cnt_ux5 = list_cnt_ux[4]
    value.cnt_ux6 = list_cnt_ux[5]
    value.cnt_ux7 = list_cnt_ux[6]
    value.cnt_ux8 = list_cnt_ux[7]
    value.cnt_ux9 = list_cnt_ux[8]
    value.cnt_ux10 = list_cnt_ux[9]
    value.cnt_ux11 = list_cnt_ux[10]
    value.cnt_ux12 = list_cnt_ux[11]
    value.cnt_ux13 = list_cnt_ux[12]
    value.cnt_ux14 = list_cnt_ux[13]
    value.cnt_ux15 = list_cnt_ux[14]
    value.cnt_ux16 = list_cnt_ux[15]
    value.cnt_ux17 = list_cnt_ux[16]
   
    
    value.cnt_uy1 = list_cnt_uy[0]
    value.cnt_uy2 = list_cnt_uy[1]
    value.cnt_uy3 = list_cnt_uy[2]
    value.cnt_uy4 = list_cnt_uy[3]
    value.cnt_uy5 = list_cnt_uy[4]
    value.cnt_uy6 = list_cnt_uy[5]
    value.cnt_uy7 = list_cnt_uy[6]
    value.cnt_uy8 = list_cnt_uy[7]
    value.cnt_uy9 = list_cnt_uy[8]
    value.cnt_uy10 = list_cnt_uy[9]
    value.cnt_uy11 = list_cnt_uy[10]
    value.cnt_uy12 = list_cnt_uy[11]
    value.cnt_uy13 = list_cnt_uy[12]
    value.cnt_uy14 = list_cnt_uy[13]
    value.cnt_uy15 = list_cnt_uy[14]
    value.cnt_uy16 = list_cnt_uy[15]
    value.cnt_uy17 = list_cnt_uy[16]


    value.angle_up1 = list_angle_up[0]
    value.angle_up2 = list_angle_up[1]
    value.angle_up3 = list_angle_up[2]
    value.angle_up4 = list_angle_up[3]
    value.angle_up5 = list_angle_up[4]
    value.angle_up6 = list_angle_up[5]
    value.angle_up7 = list_angle_up[6]
    value.angle_up8 = list_angle_up[7]
    value.angle_up9 = list_angle_up[8]
    value.angle_up10 = list_angle_up[9]
    value.angle_up11 = list_angle_up[10]
    value.angle_up12 = list_angle_up[11]
    value.angle_up13 = list_angle_up[12]
    value.angle_up14 = list_angle_up[13]
    value.angle_up15 = list_angle_up[14]
    value.angle_up16 = list_angle_up[15]
    value.angle_up17 = list_angle_up[16]
    
    
    value.area_up1 = list_area_up[0]
    value.area_up2 = list_area_up[1]
    value.area_up3 = list_area_up[2]
    value.area_up4 = list_area_up[3]
    value.area_up5 = list_area_up[4]
    value.area_up6 = list_area_up[5]
    value.area_up7 = list_area_up[6]
    value.area_up8 = list_area_up[7]
    value.area_up9 = list_area_up[8]
    value.area_up10 = list_area_up[9]
    value.area_up11 = list_area_up[10]
    value.area_up12 = list_area_up[11]
    value.area_up13 = list_area_up[12]
    value.area_up14 = list_area_up[13]
    value.area_up15 = list_area_up[14]
    value.area_up16 = list_area_up[15]
    value.area_up17 = list_area_up[16]
    

    value.angleflag_up1 = list_angleflag_up[0]
    value.angleflag_up2 = list_angleflag_up[1]
    value.angleflag_up3 = list_angleflag_up[2]
    value.angleflag_up4 = list_angleflag_up[3]
    value.angleflag_up5 = list_angleflag_up[4]
    value.angleflag_up6 = list_angleflag_up[5]
    value.angleflag_up7 = list_angleflag_up[6]
    value.angleflag_up8 = list_angleflag_up[7]
    value.angleflag_up9 = list_angleflag_up[8]
    value.angleflag_up10 = list_angleflag_up[9]
    value.angleflag_up11 = list_angleflag_up[10]
    value.angleflag_up12 = list_angleflag_up[11]
    value.angleflag_up13 = list_angleflag_up[12]
    value.angleflag_up14 = list_angleflag_up[13]
    value.angleflag_up15 = list_angleflag_up[14]
    value.angleflag_up16 = list_angleflag_up[15]
    value.angleflag_up17 = list_angleflag_up[16]
    
    #------------------------------------------------------------

    value.cnt_wx1 = list_cnt_wx[0]
    value.cnt_wx2 = list_cnt_wx[1]
    value.cnt_wx3 = list_cnt_wx[2]
    value.cnt_wx4 = list_cnt_wx[3]
    value.cnt_wx5 = list_cnt_wx[4]
    value.cnt_wx6 = list_cnt_wx[5]
    value.cnt_wx7 = list_cnt_wx[6]
    value.cnt_wx8 = list_cnt_wx[7]
    value.cnt_wx9 = list_cnt_wx[8]
    value.cnt_wx10 = list_cnt_wx[9]
    value.cnt_wx11 = list_cnt_wx[10]
    value.cnt_wx12 = list_cnt_wx[11]
    value.cnt_wx13 = list_cnt_wx[12]
    value.cnt_wx14 = list_cnt_wx[13]
    value.cnt_wx15 = list_cnt_wx[14]
    value.cnt_wx16 = list_cnt_wx[15]
    value.cnt_wx17 = list_cnt_wx[16]
    value.cnt_wx18 = list_cnt_wx[17]
   
    
    value.cnt_wy1 = list_cnt_wy[0]
    value.cnt_wy2 = list_cnt_wy[1]
    value.cnt_wy3 = list_cnt_wy[2]
    value.cnt_wy4 = list_cnt_wy[3]
    value.cnt_wy5 = list_cnt_wy[4]
    value.cnt_wy6 = list_cnt_wy[5]
    value.cnt_wy7 = list_cnt_wy[6]
    value.cnt_wy8 = list_cnt_wy[7]
    value.cnt_wy9 = list_cnt_wy[8]
    value.cnt_wy10 = list_cnt_wy[9]
    value.cnt_wy11 = list_cnt_wy[10]
    value.cnt_wy12 = list_cnt_wy[11]
    value.cnt_wy13 = list_cnt_wy[12]
    value.cnt_wy14 = list_cnt_wy[13]
    value.cnt_wy15 = list_cnt_wy[14]
    value.cnt_wy16 = list_cnt_wy[15]
    value.cnt_wy18 = list_cnt_wy[17]


    value.angle_width1 = list_angle_width[0]
    value.angle_width2 = list_angle_width[1]
    value.angle_width3 = list_angle_width[2]
    value.angle_width4 = list_angle_width[3]
    value.angle_width5 = list_angle_width[4]
    value.angle_width6 = list_angle_width[5]
    value.angle_width7 = list_angle_width[6]
    value.angle_width8 = list_angle_width[7]
    value.angle_width9 = list_angle_width[8]
    value.angle_width10 = list_angle_width[9]
    value.angle_width11 = list_angle_width[10]
    value.angle_width12 = list_angle_width[11]
    value.angle_width13 = list_angle_width[12]
    value.angle_width14 = list_angle_width[13]
    value.angle_width15 = list_angle_width[14]
    value.angle_width16 = list_angle_width[15]
    value.angle_width17 = list_angle_width[16]
    value.angle_width18 = list_angle_width[17]
    
    
    value.area_width1 = list_area_width[0]
    value.area_width2 = list_area_width[1]
    value.area_width3 = list_area_width[2]
    value.area_width4 = list_area_width[3]
    value.area_width5 = list_area_width[4]
    value.area_width6 = list_area_width[5]
    value.area_width7 = list_area_width[6]
    value.area_width8 = list_area_width[7]
    value.area_width9 = list_area_width[8]
    value.area_width10 = list_area_width[9]
    value.area_width11 = list_area_width[10]
    value.area_width12 = list_area_width[11]
    value.area_width13 = list_area_width[12]
    value.area_width14 = list_area_width[13]
    value.area_width15 = list_area_width[14]
    value.area_width16 = list_area_width[15]
    value.area_width17 = list_area_width[16]
    value.area_width18 = list_area_width[17]
    

    value.angleflag_width1 = list_angleflag_width[0]
    value.angleflag_width2 = list_angleflag_width[1]
    value.angleflag_width3 = list_angleflag_width[2]
    value.angleflag_width4 = list_angleflag_width[3]
    value.angleflag_width5 = list_angleflag_width[4]
    value.angleflag_width6 = list_angleflag_width[5]
    value.angleflag_width7 = list_angleflag_width[6]
    value.angleflag_width8 = list_angleflag_width[7]
    value.angleflag_width9 = list_angleflag_width[8]
    value.angleflag_width10 = list_angleflag_width[9]
    value.angleflag_width11 = list_angleflag_width[10]
    value.angleflag_width12 = list_angleflag_width[11]
    value.angleflag_width13 = list_angleflag_width[12]
    value.angleflag_width14 = list_angleflag_width[13]
    value.angleflag_width15 = list_angleflag_width[14]
    value.angleflag_width16 = list_angleflag_width[15]
    value.angleflag_width17 = list_angleflag_width[16]
    value.angleflag_width18 = list_angleflag_width[17]



    

           
    rospy.loginfo(value)
    pub.publish(value)
    #rate.sleep()
    return list_cnt_ux





class tak_pic_pla():
    def __init__(self,up_x,up_y,width_x,width_y):
        self.up_x = up_x    
        self.up_y = up_y
        self.width_x = width_x
        self.width_y = width_y
pic = tak_pic_pla( -18.4 , 52.8 , 25.6 , 28.8 )

class Screen_cut():            #螢幕切割
    def __init__(self,y1,y2,x1,x2):
        self.y1 = y1    
        self.y2 = y2
        self.x1 = x1
        self.x2 = x2
Screen_cutter = Screen_cut(0,1080,0,1920)

class real_center_up():           #長邊座標補償位置
    def __init__(self,y1,y2,y3,y4,y5,y6,x1,x2,x3,x4,x5,x6):
        self.y1 = y1    
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        self.y6 = y6
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
rcu = real_center_up(48.55 ,52.85 , 57.05 , 61.2 , 65.3 , 69.4 , -30.85 , -25.1 , -19.05 , -13.2 , -7.3 , -1.25 )

class real_center_width():         #寬邊座標補償位置
    def __init__(self,y1,y2,y3,y4,y5,y6,x1,x2,x3,x4,x5,x6):
        self.y1 = y1    
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        self.y6 = y6
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
rcw = real_center_width(25.35 , 29.65 , 33.9 , 38.05 , 42.25 , 46.4 , 2.35 , 8.35 , 14.35 , 20.35 , 26.05 , 32.0 )

class ori_coordinate():            #原圖拼圖位置切割
    def __init__(self,y1,y2,y3,y4,y5,y6,x1,x2,x3,x4,x5,x6,x7,x8): 
        self.y1 = y1    
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4
        self.y5 = y5
        self.y6 = y6
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        self.x7 = x7
        self.x8 = x8
oc = ori_coordinate(34,250,444,641,830,1029,341,535,732,937,1117,1318,1513,1727)  
####-------------------------list for strategy ------------------########

action = 0

cup_color = ""    #decide what's color that the cup is  

down_stop_flag = 1 #Arm stop when sucker already suck something
count_stop = 0    #decide what times need to run

pressure_info = ""

check_Arm_idle = 0
suck_state = 0

is_move = False          # 判斷時間 避免程式提前
start_time = time.time()
#for test

list_cnt_ux =[]     #長邊座標補償後座標
list_cnt_uy =[]

list_cnt_wx =[]     #寬邊座標補償後座標
list_cnt_wy =[]

list_angle_up = []  #長邊算出後角度差
list_area_up = []   #長邊屬於第幾塊拼圖
list_angleflag_up = []   #判斷是否要提前轉90
                         # 1 = 轉90
                         # 2 = 轉-90


list_angle_width = []  #寬邊算出後角度差
list_area_width = []   #寬邊屬於第幾塊拼圖
list_angleflag_width = []   #判斷是否要提前轉90
                            # 1 = 轉90
                            # 2 = 轉-90

list_c_up = []          #長邊圈出圓心 座標
list_c_width = []       #寬邊圈出圓心 座標

list_c_again = []       #再次圈出圓心 座標

####-------------------------開啟realsense------------------########
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

# #     # Start streaming
cfg = pipeline.start(config)
    
iteration = 0
preset = 0
preset_name = ''

cap = cv2.VideoCapture(0)
####--------------------------------------------------------########
class background_function():
    def displayIMG(self, img, windowName):            #照片顯示
    
        cv2.namedWindow( windowName, cv2.WINDOW_NORMAL )
    
        cv2.resizeWindow(windowName, 600, 600)
    
        cv2.imshow(windowName, img)
        cv2.waitKey()
    
    def take_pic(self):                             #拍照
        try:
            while(True):
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                color_image = np.asanyarray(color_frame.get_data())
                images = color_image
    
                # Show images
                cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                ret, RealSense = cap.read()
                cv2.imshow('RealSense', images)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('t'):
                    cv2.imwrite('/home/user/00.png',images)
                    #result = False
                    em = cv2.imread('/home/user/00.png')
                    #cv2.resize(em,(600,800))  
                    bak_fun.displayIMG(em,'hoby')
    
            
            
                #cv2.imshow('frame', frame)
                elif key & 0xFF == ord('q')or key ==27:
                    return images
                    break
        finally:
            #cap.release()
            cv2.destroyAllWindows()
            #Stop streaming
            #pipeline.stop()   
    
    def callback_yolo_receive(self,data):
        global temp_label
        temp_label = data.data #receive label to temp
    
   
bak_fun = background_function()



class function_identify():  #辨識程式
    def Contours_bounding_box_up(self,img):   #長邊圈重心
        global cun_center
        cun_center=0
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),125, 255, cv2.THRESH_BINARY)
    
        aa, ctrs, hie = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in ctrs:
            x, y, w, h = cv2.boundingRect(c)
            if (w > 130 and h >130 and w <230 and h <230):
                # get the min area rect
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)
    
                list_c_up.append(center)
                cun_center = cun_center + 1
            else:
                continue
    
        print("center",list_c_up)
        
    
        print(len(ctrs))
        # cv2.drawContours(img, ctrs, -1, (255, 255, 0), 1)
        cv2.waitKey()
        cv2.destroyAllWindows()
        return cun_center
      
    def Contours_bounding_box_width(self,img):    #寬邊圈重心
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),125, 255, cv2.THRESH_BINARY)
    
        aa, ctrs, hie = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in ctrs:
            x, y, w, h = cv2.boundingRect(c)
            if (w > 130 and h >130 and w <230 and h <230):
                # get the min area rect
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                # finally, get the min enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)
    
                list_c_width.append(center)
            else:
                continue
        print("center",list_c_width)
        
    
        print(len(ctrs))
        # cv2.drawContours(img, ctrs, -1, (255, 255, 0), 1)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def Contours_bounding_box_again(self,img):    #再次圈重心
        ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),125, 255, cv2.THRESH_BINARY)
    
        aa, ctrs, hie = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for c in ctrs:
            x, y, w, h = cv2.boundingRect(c)
            if (w > 130 and h >130 and w <230 and h <230):
                # get the min area rect
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                # convert all coordinates floating point values to int
                box = np.int0(box)
                # finally, get the min enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(c)
                # convert all values to int
                center = (int(x), int(y))
                radius = int(radius)
    
                list_c_again.append(center)
            
            else:
                continue
        print("center",list_c_again)
        
    
        print(len(ctrs))
        # cv2.drawContours(img, ctrs, -1, (255, 255, 0), 1)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def angle_detect_up(self):   #辨識長邊
        tX=0
        tY=0
        mode = 1
        img = cv2.imread('/home/jack/Desktop/two.png')   #原始拼圖
        img_item = img
        ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY) 
        fun_iden.Contours_bounding_box_up(img_item) # 圈個別物件用重心
        
        img_ori = cv2.imread('/home/jack/Desktop/test290.png',0)  #辨識用照片
        
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower = np.array([35,43,46])          #濾掉鮮綠色
        upper = np.array([88,255,255])
        mask = cv2.inRange(hsv,lower,upper)
        ret,thresh2 = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
        edge_copy = thresh2[Screen_cutter.y1:Screen_cutter.y2,Screen_cutter.x1:Screen_cutter.x2]
        
        #開始尋找物件
        (_,cnts,_) = cv2.findContours(edge_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        
    
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            if(h >60 and w > 60 and w <330 and h<330):   #篩選過大過小框框         
                for i in range (17) :    #篩選出正確圓心進入編號
                    if(abs( (x+w/2) - list_c_up[i][0] ) < 100 ):  #比較取絕對值
                        if(abs( (y+h/2) - list_c_up[i][1] ) < 100 ):
                            tX = list_c_up[i][0]
                            tY = list_c_up[i][1]
                            print("tX:",tX)
                            print("tY:",tY)
                        else: 
                            continue
                    else:   
                        continue
    
                if(mode == 1): #框出最小正方形
                    cv2.rectangle(edge_copy, (x-2, y-2), (x+w+2, y+h+2), (255,0,0), 2)
                    items = img_item[y-5:(y + h)+5, x-5:(x + w)+5]
    
                elif(mode == 2): #同上但框框會旋轉來圈出最小正方形
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    print(box)
                    cv2.drawContours(edge_copy, [box], 0, (255, 0, 0), 2)
                    items = edged[y-5:(y + h)+5, x-5:(x + w)+5]
    
                if (h >60 and w > 60): #大於這個範圍的才會認定為一塊拼圖
                    items= cv2.cvtColor(items, cv2.COLOR_BGR2GRAY)
                    nX,nY = fun_iden.new_fit(tX,tY)
                    #座標轉換(linear)
                    real_x = -22.384903 + 1.000154*nX   # + 1.2
                    real_y =  58.850340 + -1.001752*nY  # + 0.8
                    fun_iden.compensation_position_up(real_x,real_y)
                    fun_iden.sift_angle_up(img_ori,items) 
                    print('len_up::::',len(list_cnt_ux))
                           #SIFI flann match
    
    def angle_detect_width(self):   #辨識寬邊
        mode = 1
        img = cv2.imread('/home/jack/Desktop/one.png')  #原始拼圖
        img_item = img
        ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY) 
        fun_iden.Contours_bounding_box_width(img_item)  # 圈個別物件用重心
        
        img_ori = cv2.imread('/home/jack/Desktop/test290.png',0)  #辨識用照片
        
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        lower = np.array([35,43,46])   #濾掉鮮綠色
        upper = np.array([88,255,255])
        mask = cv2.inRange(hsv,lower,upper)
        ret,thresh2 = cv2.threshold(mask,0,255,cv2.THRESH_BINARY_INV)
        edge_copy = thresh2[Screen_cutter.y1:Screen_cutter.y2,Screen_cutter.x1:Screen_cutter.x2]  #切割
    
        #開始尋找物件
        (_,cnts,_) = cv2.findContours(edge_copy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        print("gogo")
        
    
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            if(h >60 and w > 60 and w <330 and h<330):  #篩選過大過小框框       
                for i in range (18) :    #篩選出正確圓心進入編號
                    if(abs( (x+w/2) - list_c_width[i][0] ) < 30 ):  #比較取絕對值
                        if(abs( (y+h/2) - list_c_width[i][1] ) < 30 ):
                            tX = list_c_width[i][0]
                            tY = list_c_width[i][1] 
                            print(tX,tY,"tx ty")
                        else: 
                            continue
                    else:   
                        continue
        
                if(mode == 1): #框出最小正方形
                    cv2.rectangle(edge_copy, (x-2, y-2), (x+w+2, y+h+2), (255,0,0), 2)
                    items = img_item[y-5:(y + h)+5, x-5:(x + w)+5]
        
                elif(mode == 2): #同上但框框會旋轉來圈出最小正方形
                    rect = cv2.minAreaRect(c)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    print(box)
                    cv2.drawContours(edge_copy, [box], 0, (255, 0, 0), 2)
                    items = edged[y-5:(y + h)+5, x-5:(x + w)+5]
        
                if (h >60 and w > 60): #大於這個範圍的才會認定為一塊拼圖
                    items= cv2.cvtColor(items, cv2.COLOR_BGR2GRAY)
                    nX,nY = fun_iden.new_fit(tX,tY)
                    #座標轉換(linear)
                    real_x = 21.633181 + 1.002529*nX  # + 0.8
                    real_y =  35.359252 - 0.999399*nY # - 0.3
                    fun_iden.compensation_position_width(real_x,real_y)   #補償座標轉換
                    fun_iden.sift_angle_width(img_ori,items)       #SIFI flann match


    def new_fit(self,X,Y):               #大地座標轉換
        c_x = 934.711791992188
        c_y = 548.315856933594
        f_x = 1379.17150878906
        f_y = 1379.71948242188
        z_world = 34
        x_world = (X - c_x) * z_world / f_x
        y_world = (Y - c_y) * z_world / f_y
    
        return x_world,y_world

    flag1 = 0 
    def compensation_position_up(self,ral_x,ral_y):       #長邊座標補償位置
        global list_cnt_ux,list_cnt_uy,flag1
        if (ral_x > rcu.x1 and ral_x < rcu.x2):
            if ( ral_y  > rcu.y1 and ral_y < rcu.y2):
                print("puzzle_area 1")
                real_x = ral_x 
                real_y = ral_y + 0.1
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcu.y2 and ral_y  < rcu.y3):
                print("puzzle_area 6") 
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y3 and ral_y < rcu.y4):
                print("puzzle_area 11")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y4 and ral_y < rcu.y5):
                print("puzzle_area 16")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y5 and ral_y < rcu.y6):
                print("puzzle_area 21")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcu.x2 and ral_x < rcu.x3):
            if ( ral_y  > rcu.y1 and ral_y < rcu.y2):
                print("puzzle_area 2")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcu.y2 and ral_y  < rcu.y3):
                print("puzzle_area 7")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y3 and ral_y < rcu.y4):
                print("puzzle_area 12")
                real_x = ral_x 
                real_y = ral_y + 0.2
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y4 and ral_y < rcu.y5):
                print("puzzle_area 17")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y5 and ral_y < rcu.y6):
                print("puzzle_area 22")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcu.x3 and ral_x < rcu.x4):
            if ( ral_y  > rcu.y1 and ral_y < rcu.y2):
                print("puzzle_area 3")
                real_x = ral_x 
                real_y = ral_y + 0.25
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcu.y2 and ral_y  < rcu.y3):
                print("puzzle_area 8")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y3 and ral_y < rcu.y4):
                print("puzzle_area 13")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y4 and ral_y < rcu.y5):
                print("puzzle_area 18")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y5 and ral_y < rcu.y6):
                print("puzzle_area 23")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcu.x4 and ral_x < rcu.x5):
            if ( ral_y  > rcu.y1 and ral_y < rcu.y2):
                print("puzzle_area 4")
                real_x = ral_x 
                real_y = ral_y  + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcu.y2 and ral_y  < rcu.y3):
                print("puzzle_area 9")
                real_x = ral_x 
                real_y = ral_y   + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y3 and ral_y < rcu.y4):
                print("puzzle_area 14")
                real_x = ral_x 
                real_y = ral_y  + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y4 and ral_y < rcu.y5):
                print("puzzle_area 19")
                real_x = ral_x 
                real_y = ral_y   + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y5 and ral_y < rcu.y6):
                print("puzzle_area 24")
                real_x = ral_x 
                real_y = ral_y  + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
        elif (ral_x > rcu.x5 and ral_x < rcu.x6):
            if ( ral_y  > rcu.y1 and ral_y < rcu.y2):
                print("puzzle_area 5")
                real_x = ral_x 
                real_y = ral_y   + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcu.y2 and ral_y  < rcu.y3):
                print("puzzle_area 10")
                real_x = ral_x 
                real_y = ral_y  + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y3 and ral_y < rcu.y4):
                print("puzzle_area 15")
                real_x = ral_x 
                real_y = ral_y    + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y4 and ral_y < rcu.y5):
                print("puzzle_area 20")
                real_x = ral_x 
                real_y = ral_y   + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcu.y5 and ral_y < rcu.y6):
                print("puzzle_area 25")
                real_x = ral_x 
                real_y = ral_y   + 0.3
                list_cnt_ux.append(real_x)
                list_cnt_uy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)  
            print( list_cnt_ux," list_cnt_ux[0]")
            
    def compensation_position_width(self,ral_x,ral_y):          #寬邊座標補償位置
        global list_cnt_wx,list_cnt_wy
        if (ral_x > rcw.x1 and ral_x < rcw.x2):
            if ( ral_y  > rcw.y1 and ral_y < rcw.y2):
                print("puzzle_area 1")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcw.y2 and ral_y  < rcw.y3):
                print("puzzle_area 6")
                real_x = ral_x
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y3 and ral_y < rcw.y4):
                print("puzzle_area 11")
                real_x = ral_x 
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y4 and ral_y < rcw.y5):
                print("puzzle_area 16")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y5 and ral_y < rcw.y6):
                print("puzzle_area 21")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcw.x2 and ral_x < rcw.x3):
            if ( ral_y  > rcw.y1 and ral_y < rcw.y2):
                print("puzzle_area 2")
                real_x = ral_x - 0.1
                real_y = ral_y - 0.2
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcw.y2 and ral_y  < rcw.y3):
                print("puzzle_area 7")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y3 and ral_y < rcw.y4):
                print("puzzle_area 12")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y4 and ral_y < rcw.y5):
                print("puzzle_area 17")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y5 and ral_y < rcw.y6):
                print("puzzle_area 22")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcw.x3 and ral_x < rcw.x4):
            if ( ral_y  > rcw.y1 and ral_y < rcw.y2):
                print("puzzle_area 3")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcw.y2 and ral_y  < rcw.y3):
                print("puzzle_area 8")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y3 and ral_y < rcw.y4):
                print("puzzle_area 13")
                real_x = ral_x - 0.3
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y4 and ral_y < rcw.y5):
                print("puzzle_area 18")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y5 and ral_y < rcw.y6):
                print("puzzle_area 23")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            else:
                list_area.append("e")
    
        elif (ral_x > rcw.x4 and ral_x < rcw.x5):
            if ( ral_y  > rcw.y1 and ral_y < rcw.y2):
                print("puzzle_area 4")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcw.y2 and ral_y  < rcw.y3):
                print("puzzle_area 9")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y3 and ral_y < rcw.y4):
                print("puzzle_area 14")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y4 and ral_y < rcw.y5):
                print("puzzle_area 19")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y5 and ral_y < rcw.y6):
                print("puzzle_area 24")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
        elif (ral_x > rcw.x5 and ral_x < rcw.x6):
            if ( ral_y  > rcw.y1 and ral_y < rcw.y2):
                print("puzzle_area 5")
                real_x = ral_x 
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
    
            elif ( ral_y > rcw.y2 and ral_y  < rcw.y3):
                print("puzzle_area 10")
                real_x = ral_x 
                real_y = ral_y 
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y3 and ral_y < rcw.y4):
                print("puzzle_area 15")
                real_x = ral_x 
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y4 and ral_y < rcw.y5):
                print("puzzle_area 20")
                real_x = ral_x 
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
            elif ( ral_y > rcw.y5 and ral_y < rcw.y6):
                print("puzzle_area 25")
                real_x = ral_x
                real_y = ral_y
                list_cnt_wx.append(real_x)
                list_cnt_wy.append(real_y)
                print("ral_x",real_x)
                print("ral_y",real_y)
            
    def sift_angle_up(self,img1,img2):         #長邊 拼圖辨識
        list_x1 = []      #save object center
        list_y1 = []
        MIN_MATCH_COUNT = 7 #设置最低特征点匹配数量为10
        box_in_sence = img1 # queryImage
        box = img2
        
        # 创建sift特征检测器
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(box,None)
        kp2, des2 = sift.detectAndCompute(box_in_sence,None)
        
        # 匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        goodMatches = []
        
        # 筛选出好的描述子
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append(m)
                
        obj_pts, scene_pts = [], []
        
        # 单独保存 obj 和 scene 好的点位置
        print ("goodMatches up : ",len(goodMatches))
        
        if len(goodMatches)>MIN_MATCH_COUNT:
            # 获取关键点的坐标
            obj_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            scene_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            #计算变换矩阵和MASK
            M, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = box.shape
    
            # pts 為box照片的四個角落座標
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            
            X_abs = int((dst[0][0]+dst[2][0])/2)
            Y_abs = int((dst[0][1]+dst[2][1])/2)
            cv2.circle(box_in_sence, (X_abs, Y_abs), 10, (1, 227, 254), -1) 
            list_x1.append(X_abs)
            list_y1.append(Y_abs)
            fun_iden.area_detection_up(list_x1,list_y1)
    
            # 求角度
            X1 = pts[2][0][0] - pts[0][0][0]
            Y1 = pts[2][0][1] - pts[0][0][1]
    
            X2 = dst[2][0] - dst[0][0]
            Y2 = dst[2][1] - dst[0][1]
    
            angle1 = np.rad2deg(np.arctan2(Y1,X1))
            angle2 = np.rad2deg(np.arctan2(Y2,X2))
            angle_diff = angle2 - angle1
        
        #角度轉換讓手臂不轉超過120 以免進入手臂奇異點 
        if (angle_diff > 180):
            angle_diff = angle_diff - 180
            if (angle_diff > 90):
                angle_diff = 180 - angle_diff
                list_angle_up.append(angle_diff)
                Angle_flag = 0
                list_angleflag_up.append(Angle_flag)
            else:
                angle_diff = 90 - angle_diff
                list_angle_up.append(angle_diff)
                Angle_flag = 2
                list_angleflag_up.append(Angle_flag)
    
        elif (angle_diff > 90 and angle_diff < 180):
            angle_diff = angle_diff - 90
            angle_diff = angle_diff * -1
            list_angle_up.append(angle_diff)
            Angle_flag = 1
            list_angleflag_up.append(Angle_flag)
    
        elif (angle_diff < -180):
            angle_diff = angle_diff + 180
            if(angle_diff > -90):
                angle_diff = angle_diff + 90
                angle_diff = angle_diff * -1
                list_angle_up.append(angle_diff)
                Angle_flag = 1
                list_angleflag_up.append(Angle_flag)
            else:
                angle_diff = angle_diff 
                list_angle_up.append(angle_diff)
                Angle_flag = 0
                list_angleflag_up.append(Angle_flag)
    
        elif (angle_diff < -90 and angle_diff > -180):
            angle_diff = angle_diff + 90
            angle_diff = angle_diff * -1
            list_angle_up.append(angle_diff)    
            Angle_flag = 2
            list_angleflag_up.append(Angle_flag)
        else:
            angle_diff = angle_diff * -1
            list_angle_up.append(angle_diff)
            Angle_flag = 0
            list_angleflag_up.append(Angle_flag)
    
            # 加上偏移量
        for i in range(4):
            dst[i][0] += w
        else:
            print( "Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))
            matchesMask = None
        
        draw_params = dict(singlePointColor=None,
                           matchesMask=matchesMask, 
                           flags=2)
        result = cv2.drawMatches( box, kp1, box_in_sence, kp2, goodMatches, None,**draw_params)
        cv2.polylines(result, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)

        
    
    def sift_angle_width(self,img1,img2):      #寬邊 拼圖辨識
        list_x1 = []      #save object center
        list_y1 = []
        MIN_MATCH_COUNT = 7 #设置最低特征点匹配数量为10
        box_in_sence = img1 # queryImage
        box = img2
        
        # 创建sift特征检测器
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(box,None)
        kp2, des2 = sift.detectAndCompute(box_in_sence,None)
        
        # 匹配
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        goodMatches = []
        
        # 筛选出好的描述子
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                goodMatches.append(m)
                
        obj_pts, scene_pts = [], []
        # 单独保存 obj 和 scene 好的点位置
        print('len_width::::',len(list_cnt_wx))
        if len(goodMatches)>MIN_MATCH_COUNT:
            # 获取关键点的坐标
            obj_pts = np.float32([ kp1[m.queryIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            scene_pts = np.float32([ kp2[m.trainIdx].pt for m in goodMatches ]).reshape(-1,1,2)
            #计算变换矩阵和MASK
            M, mask = cv2.findHomography(obj_pts, scene_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = box.shape
    
            # pts 為box照片的四個角落座標
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)  
            dst = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            
            X_abs = int((dst[0][0]+dst[2][0])/2)
            Y_abs = int((dst[0][1]+dst[2][1])/2)
            cv2.circle(box_in_sence, (X_abs, Y_abs), 10, (1, 227, 254), -1) 
            list_x1.append(X_abs)
            list_y1.append(Y_abs)
            fun_iden.area_detection_width(list_x1,list_y1)
    
            # 求角度
            X1 = pts[2][0][0] - pts[0][0][0]
            Y1 = pts[2][0][1] - pts[0][0][1]
    
            X2 = dst[2][0] - dst[0][0]
            Y2 = dst[2][1] - dst[0][1]
    
            angle1 = np.rad2deg(np.arctan2(Y1,X1))
            angle2 = np.rad2deg(np.arctan2(Y2,X2))
            angle_diff = angle2 - angle1
        
        #角度轉換讓手臂不轉超過120 以免進入手臂奇異點 
        if (angle_diff > 180):
            angle_diff = angle_diff - 180
            if (angle_diff > 90):
                angle_diff = 180 - angle_diff
                list_angle_width.append(angle_diff)
                Angle_flag = 0
                list_angleflag_width.append(Angle_flag)
            else:
                angle_diff = 90 - angle_diff
                list_angle_width.append(angle_diff)
                Angle_flag = 2
                list_angleflag_width.append(Angle_flag)
    
        elif (angle_diff > 90 and angle_diff < 180):
            angle_diff = angle_diff - 90
            angle_diff = angle_diff * -1
            list_angle_width.append(angle_diff)
            Angle_flag = 1
            list_angleflag_width.append(Angle_flag)
    
        elif (angle_diff < -180):
            angle_diff = angle_diff + 180
            if(angle_diff > -90):
                angle_diff = angle_diff + 90
                angle_diff = angle_diff * -1
                list_angle_width.append(angle_diff)
                Angle_flag = 1
                list_angleflag_width.append(Angle_flag)
            else:
                angle_diff = angle_diff 
                list_angle_width.append(angle_diff)
                Angle_flag = 0
                list_angleflag_width.append(Angle_flag)
    
        elif (angle_diff < -90 and angle_diff > -180):
            angle_diff = angle_diff + 90
            angle_diff = angle_diff * -1
            list_angle_width.append(angle_diff)    
            Angle_flag = 2
            list_angleflag_width.append(Angle_flag)
        else:
            angle_diff = angle_diff * -1
            list_angle_width.append(angle_diff)
            Angle_flag = 0
            list_angleflag_width.append(Angle_flag)
    
            # 加上偏移量
        for i in range(4):
            dst[i][0] += w
        else:
            print( "Not enough matches are found - %d/%d" % (len(goodMatches),MIN_MATCH_COUNT))
            matchesMask = None
        
        draw_params = dict(singlePointColor=None,
                           matchesMask=matchesMask, 
                           flags=2)
        result = cv2.drawMatches( box, kp1, box_in_sence, kp2, goodMatches, None,**draw_params)
        cv2.polylines(result, [np.int32(dst)], True, (0, 0, 255), 3, cv2.LINE_AA)
       
    def area_detection_up(self,img_x,img_y):    #長邊屬於第幾塊拼圖
        if (img_x[0] > oc.x1 and img_x[0] < oc.x2):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(1)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(8)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(15)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(22)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(29)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x2 and img_x[0] < oc.x3):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(2)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(9)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(16)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(23)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(30)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x3 and img_x[0] < oc.x4):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(3)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(10)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(17)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(24)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(31)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x4 and img_x[0] < oc.x5):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(4)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(11)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(18)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(25)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(32)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x5 and img_x[0] < oc.x6):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(5)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(12)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(19)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(26)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(33)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x6 and img_x[0] < oc.x7):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(6)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(13)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(20)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(27)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(34)
    
            else:
                list_area_up.append(e)
    
        elif (img_x[0] > oc.x7 and img_x[0] < oc.x8):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_up.append(7)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_up.append(14)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_up.append(21)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_up.append(28)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_up.append(35)
    
            else:
                list_area_up.append(e)
    
    def area_detection_width(self,img_x,img_y):     #寬邊屬於第幾塊拼圖
        if (img_x[0] > oc.x1 and img_x[0] < oc.x2):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(1)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(8)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(15)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(22)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(29)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x2 and img_x[0] < oc.x3):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(2)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(9)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(16)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(23)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(30)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x3 and img_x[0] < oc.x4):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(3)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(10)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(17)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(24)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(31)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x4 and img_x[0] < oc.x5):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(4)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(11)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(18)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(25)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(32)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x5 and img_x[0] < oc.x6):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(5)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(12)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(19)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(26)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(33)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x6 and img_x[0] < oc.x7):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(6)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(13)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(20)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(27)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(34)
    
            else:
                list_area_width.append(e)
    
        elif (img_x[0] > oc.x7 and img_x[0] < oc.x8):
            if (img_y[0] > oc.y1 and img_y[0] < oc.y2):
                list_area_width.append(7)
                
            elif (img_y[0] > oc.y2 and img_y[0] < oc.y3):
                list_area_width.append(14)
            
            elif (img_y[0] > oc.y3 and img_y[0] < oc.y4):
                list_area_width.append(21)
            
            elif (img_y[0] > oc.y4 and img_y[0] < oc.y5):
                list_area_width.append(28)
            
            elif (img_y[0] > oc.y5 and img_y[0] < oc.y6):
                list_area_width.append(35)
    
            else:
                list_area_width.append(e)

fun_iden = function_identify()


# man_fun = main_function()
##-------------strategy end ------------
def myhook():
    print ("shutdown time!")

if __name__ == '__main__':
    # cv2.destroyAllWindows()
    pub = rospy.Publisher('apple', haha,queue_size=100)
    rospy.init_node('hiwin_nt1')
    rate = rospy.Rate(10)
    # pos = haha()

    argv = rospy.myargv()
    # pub  = rospy.Publisher('africa2', vision ,queue_size=100)
    #images = bak_fun.take_pic()
    #cv2.imwrite('/home/user/jack/puzzle_pic_width.png',images)
    #images = bak_fun.take_pic()
    #cv2.imwrite('/home/user/BB/puzzle_pic_up.png',images)
    # for i in range(100):
    #     talker(1,2,3,4,5)
    time.sleep(3)
    images = bak_fun.take_pic()
    cv2.imwrite('/home/jack/Desktop/one.png',images)
    time.sleep(10)
    images = bak_fun.take_pic()
    cv2.imwrite('/home/jack/Desktop/two.png',images)
    up_decide = threading.Thread(target = fun_iden.angle_detect_up)  #長邊多執行緒
    up_decide.start()
    #width_decide = threading.Thread(target = fun_iden.angle_detect_width)    #寬邊多執行緒
    #width_decide.start()
    time.sleep(30)
    #while(1):
    if(len(list_cnt_ux)==17):
        talker()
        print('ok~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    
    #while(1):
        #talker(1,2,3,4,5)
    #while (1) : print("1")
    
    # value = vision()
    # value.a1 = list_cnt_ux[0]
    # value.a2 = list_cnt_ux[1]
    # value.a3 = list_cnt_ux[2]
    # value.a4 = list_cnt_ux[3]
    # value.a5 = list_cnt_ux[4]
    # value.a6 = list_cnt_ux[5]
    # value.a7 = list_cnt_ux[6]
    # value.a8 = list_cnt_ux[7]
    # value.a9 = list_cnt_ux[8]
    # value.a10 = list_cnt_ux[9]
    # value.a11 = list_cnt_ux[10]
    # value.a12 = list_cnt_ux[11]
    # value.a13 = list_cnt_ux[12]
    # value.a14 = list_cnt_ux[13]
    # value.a15 = list_cnt_ux[14]
    # value.a16 = list_cnt_ux[15]
    # value.a17 = list_cnt_ux[16]
    # value.a18 = list_cnt_ux[17]
    # value.a19 = list_cnt_ux[18]
    # value.a20 = list_cnt_ux[19]
    # value.a21 = list_cnt_ux[20]
    # value.a22 = list_cnt_ux[21]
    # value.a23 = list_cnt_ux[22]
    # value.a24 = list_cnt_ux[23]
    # value.a25 = list_cnt_ux[24]
    # value.a26 = list_cnt_ux[25]
    # value.a27 = list_cnt_ux[26]
    # value.a28 = list_cnt_ux[27]
    # value.a29 = list_cnt_ux[28]
    # value.a30 = list_cnt_ux[29]
    # value.a31 = list_cnt_ux[30]
    # value.a32 = list_cnt_ux[31]
    # value.a33 = list_cnt_ux[32]
    # value.a34 = list_cnt_ux[33]
    # value.a35 = list_cnt_ux[34]
    # value.b1 = list_cnt_uy[0]
    # value.b2 = list_cnt_uy[1]
    # value.b3 = list_cnt_uy[2]
    # value.b4 = list_cnt_uy[3]
    # value.b5 = list_cnt_uy[4]
    # value.b6 = list_cnt_uy[5]
    # value.b7 = list_cnt_uy[6]
    # value.b8 = list_cnt_uy[7]
    # value.b9 = list_cnt_uy[8]
    # value.b10 = list_cnt_uy[9]
    # value.b11 = list_cnt_uy[10]
    # value.b12 = list_cnt_uy[11]
    # value.b13 = list_cnt_uy[12]
    # value.b14 = list_cnt_uy[13]
    # value.b15 = list_cnt_uy[14]
    # value.b16 = list_cnt_uy[15]
    # value.b17 = list_cnt_uy[16]
    # value.b18 = list_cnt_uy[17]
    # value.b19 = list_cnt_uy[18]
    # value.b20 = list_cnt_uy[19]
    # value.b21 = list_cnt_uy[20]
    # value.b22 = list_cnt_uy[21]
    # value.b23 = list_cnt_uy[22]
    # value.b24 = list_cnt_uy[23]
    # value.b25 = list_cnt_uy[24]
    # value.b26 = list_cnt_uy[25]
    # value.b27 = list_cnt_uy[26]
    # value.b28 = list_cnt_uy[27]
    # value.b29 = list_cnt_uy[28]
    # value.b30 = list_cnt_uy[29]
    # value.b31 = list_cnt_uy[30]
    # value.b32 = list_cnt_uy[31]
    # value.b33 = list_cnt_uy[32]
    # value.b34 = list_cnt_uy[33]
    # value.b35 = list_cnt_uy[34]
    # value.c1 = list_cnt_wx[0]
    # value.c2 = list_cnt_wx[1]
    # value.c3 = list_cnt_wx[2]
    # value.c4 = list_cnt_wx[3]
    # value.c5 = list_cnt_wx[4]
    # value.c6 = list_cnt_wx[5]
    # value.c7 = list_cnt_wx[6]
    # value.c8 = list_cnt_wx[7]
    # value.c9 = list_cnt_wx[8]
    # value.c10 = list_cnt_wx[9]
    # value.c11 = list_cnt_wx[10]
    # value.c12 = list_cnt_wx[11]
    # value.c13 = list_cnt_wx[12]
    # value.c14 = list_cnt_wx[13]
    # value.c15 = list_cnt_wx[14]
    # value.c16 = list_cnt_wx[15]
    # value.c17 = list_cnt_wx[16]
    # value.c18 = list_cnt_wx[17]
    # value.c19 = list_cnt_wx[18]
    # value.c20 = list_cnt_wx[19]
    # value.c21 = list_cnt_wx[20]
    # value.c22 = list_cnt_wx[21]
    # value.c23 = list_cnt_wx[22]
    # value.c24 = list_cnt_wx[23]
    # value.c25 = list_cnt_wx[24]
    # value.c26 = list_cnt_wx[25]
    # value.c27 = list_cnt_wx[26]
    # value.c28 = list_cnt_wx[27]
    # value.c29 = list_cnt_wx[28]
    # value.c30 = list_cnt_wx[29]
    # value.c31 = list_cnt_wx[30]
    # value.c32 = list_cnt_wx[31]
    # value.c33 = list_cnt_wx[32]
    # value.c34 = list_cnt_wx[33]
    # value.c35 = list_cnt_wx[34]
    # value.d1 = list_cnt_wy[0]
    # value.d2 = list_cnt_wy[1]
    # value.d3 = list_cnt_wy[2]
    # value.d4 = list_cnt_wy[3]
    # value.d5 = list_cnt_wy[4]
    # value.d6 = list_cnt_wy[5]
    # value.d7 = list_cnt_wy[6]
    # value.d8 = list_cnt_wy[7]
    # value.d9 = list_cnt_wy[8]
    # value.d10 = list_cnt_wy[9]
    # value.d11 = list_cnt_wy[10]
    # value.d12 = list_cnt_wy[11]
    # value.d13 = list_cnt_wy[12]
    # value.d14 = list_cnt_wy[13]
    # value.d15 = list_cnt_wy[14]
    # value.d16 = list_cnt_wy[15]
    # value.d17 = list_cnt_wy[16]
    # value.d18 = list_cnt_wy[17]
    # value.d19 = list_cnt_wy[18]
    # value.d20 = list_cnt_wy[19]
    # value.d21 = list_cnt_wy[20]
    # value.d22 = list_cnt_wy[21]
    # value.d23 = list_cnt_wy[22]
    # value.d24 = list_cnt_wy[23]
    # value.d25 = list_cnt_wy[24]
    # value.d26 = list_cnt_wy[25]
    # value.d27 = list_cnt_wy[26]
    # value.d28 = list_cnt_wy[27]
    # value.d29 = list_cnt_wy[28]
    # value.d30 = list_cnt_wy[29]
    # value.d31 = list_cnt_wy[30]
    # value.d32 = list_cnt_wy[31]
    # value.d33 = list_cnt_wy[32]
    # value.d34 = list_cnt_wy[33]
    # value.d35 = list_cnt_wy[34]
    # value.flag = 1
    # pub.publish(value)

    # rospy.spin()
        # else:
        #     print("area: ",list_area_up[count_stop])
        #     print("angle: ",list_angle_up[count_stop])
    # rospy.init_node('strategy', anonymous=True)
    # GetInfoFlag = True #Test no data
    # arm_state_listener()
    # bak_fun.set_suc_pub()

    # rospy.Subscriber("raspberry_receive",String,callback_receive_sucker)

    # while 1:
    #     start_input = int(input('開始策略請按1 ,下二次判斷請按2 ,測試吸盤請按3 ,五點拍照校正請按4 , 測試路平專案請按5 , 離開請按6 : ')) #輸入開始指令

    #     if start_input == 1:
    #         while 1:
    #             frames = pipeline.wait_for_frames()
    #             color_frame = frames.get_color_frame()        
    #             color_image = np.asanyarray(color_frame.get_data())
    #             images = color_image
    #             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #             ret, RealSense = cap.read()
    #             key = cv2.waitKey(1)
    #             cv2.destroyAllWindows()
    #             man_fun.Mission_strategy_two_act()
    #             if count_stop == 50:
    #                 print(" Done! you can stop~")
    #                 break

    #     if start_input == 2:
    #         while 1:
    #             frames = pipeline.wait_for_frames()
    #             color_frame = frames.get_color_frame()    
    #             color_image = np.asanyarray(color_frame.get_data())
    #             images = color_image
    #             cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    #             ret, RealSense = cap.read()  
    #             key = cv2.waitKey(1)
    #             cv2.destroyAllWindows()
    #             man_fun.Mission_two_decide()
    #             if count_stop == 50:
    #                 print(" Done! you can stop~")
    #                 break

        
        
    #     if start_input == 6:
    #         break
        
    #     count_stop = 0

    # ArmTask.rospy.spin()
   
