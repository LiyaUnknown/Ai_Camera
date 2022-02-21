
import threading
from tkinter import *
from tkinter import messagebox
import cv2
import time
import datetime

from webob import minute
from eyes import *
from pyfirmata import Arduino , SERVO
import tkinter
import pickle

all_day_saved = []


def capture(cap_num = 0 , find_people = False , recognize_hand = False , face_recognizer_mode1 = False , face_recognizer_mode2  = False , hand_camera_movement = False , face_camera_movement = False , person_camera_movement = False , person_tracking = False ) : 
    cap = cv2.VideoCapture(cap_num)
    x , y , name , color , x1 , y1 , w , h = 0,0,0,0,0,0,0,0
    while True :
        _ , frame = cap.read()
        y_s , x_s , z = frame.shape
        if find_people == True : 
            try : 
                all_ = tuple(start_find_contact(frame))
                name = all_[0]
                x , y , w , h = all_[1] , all_[2] , all_[3] , all_[4]
                cv2.putText(frame , name , (x-10 , y-10) , cv2.FONT_HERSHEY_COMPLEX , 4 , (0 , 0 , 255) , 2)
                cv2.rectangle(frame , (x , y) , (w ,h) , (0 , 0 , 255) , 4)
            except : 
                None
        if recognize_hand == True : 
            try : 
                all_1 = tuple(hand_tracking(frame))
                x1 , y1 = all_1[0] , all_1[1]
                cv2.circle(frame , (x1 , y1) , 6 , (0 , 0 , 255) , 6)
            except : 
                None
        if face_recognizer_mode1 == True : 
            try : 
                all_3 = tuple(face_tracking(frame))
                x1 , y1 = all_3[0] , all_3[1]
                cv2.circle(frame , (x1 , y1) , 6 , (0 , 0 , 255) , 6)
            except : 
                None
        if face_recognizer_mode2 == True : 
            try : 
                all_2 = tuple(start_find_contact(frame))
                x , y , w , h = all_2[1] , all_2[2] , all_2[3] , all_2[4]
                cv2.rectangle(frame , (x , y) , (w ,h) , (0 , 0 , 255) , 4)
            except : 
                None
        if face_camera_movement == True : 
            try : 
                all_3 = tuple(face_tracking(frame))
                x1 , y1 = all_3[0] , all_3[1]
                cv2.circle(frame , (x1 , y1) , 6 , (0 , 0 , 255) , 6)
            except : 
                None
            camera_movement(x1 , y1 , 1)
        if hand_camera_movement == True : 
            try : 
                all_1 = tuple(hand_tracking(frame))
                x1 , y1 = all_1[0] , all_1[1]
                cv2.circle(frame , (x1 , y1) , 6 , (0 , 0 , 255) , 6)
                
            except : 
                None
            camera_movement(x1 , y1 , 1)
        #label , color , x , y , w , h

        if person_tracking == True :
            try : 
                all_1 = tuple(object_tracking(frame))
                name , color , x1 , y1 , w , h = all_1[0] , all_1[1] , all_1[2] , all_1[3] , all_1[4] , all_1[5]
                cv2.putText(frame , name , (x1 , y1) , cv2.FONT_HERSHEY_COMPLEX , 2 , (0,0,0) , 2)
                cv2.rectangle(frame , (x1 , y1) , (w ,h) , (0,0,0) , 4)
            except : 
                None


        cv2.imshow('cap' , frame)
        keyCode = cv2.waitKey(1)

        if cv2.getWindowProperty('cap', cv2.WND_PROP_VISIBLE) <1:
            break
    cv2.destroyAllWindows()

def ft1() : 
    capture(face_recognizer_mode1=True)
def ht() :
    capture(recognize_hand=True)
def fp() :
    capture(find_people=True)
def ft2() : 
    capture(face_recognizer_mode2=True)
def cf() : 
    capture(face_camera_movement=True)
def ch() : 
    capture(hand_camera_movement=True)
def ptc() : 
    capture(person_camera_movement=True)
def pt() : 
    capture(person_tracking=True)
def th() : 
    Take_Photo_with_hand(f'random_person_code_{random.randint(1,10000)}')

root1 = Tk()
root1.geometry('250x330')
root1.title('Security_camera')
root1.attributes('-toolwindow' , -1)

Label(root1 , text='--Testing--').place(x = 50 , y = 10)

b1 = Button(root1 , text = 'face_tracking_mode1' , command=ft1).place(x = 30 , y = 40)
b2 = Button(root1 , text = 'hand_tracking' , command=ht).place(x = 30 , y = 70)
b3 = Button(root1 , text = 'find_people' , command=fp).place(x = 30 , y = 100)
b4 = Button(root1 , text = 'face_tracking_mode2' , command=ft2).place(x = 30 , y = 130)
b5 = Button(root1 , text = 'camera_movement_with_face' , command=cf).place(x = 30 , y = 160)
b6 = Button(root1 , text = 'camera_movement_with_hand' , command=ch).place(x = 30 , y = 190)
b7 = Button(root1 , text = 'camera_movement_with_person' , command=ptc).place(x = 30 , y = 220)
b8 = Button(root1 , text = 'person_tracking' , command=pt).place(x = 30 , y = 250)
b9 = Button(root1 , text = 'Take_picture_with_hand' , command=th).place(x = 30 , y = 280)


root1.mainloop()