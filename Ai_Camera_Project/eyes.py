from pickle import TRUE
import threading
from tkinter import Frame
from tracemalloc import start
import cv2
import ntpath
import os
from cv2 import VideoCapture
from cv2 import threshold
import face_recognition
import glob
import numpy as np
import time
import mediapipe as mp
import sys
import random
from plyer import notification
from PIL import Image
from torch import layer_norm, true_divide
from transformers import Wav2Vec2ForCTC , Wav2Vec2Processor
from pyfirmata import Arduino , SERVO

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

class Take_Photo_with_hand :
    def __init__(self , file_name , show_resulte = True , notif_sended = True , cap_num = 0  ) :
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(1)
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mech = mp.solutions.face_mesh
        self.face_mesh2 = self.face_mech.FaceMesh()
        self.cap_num = cap_num
        global xe
        global ye
        cap = cv2.VideoCapture(self.cap_num)
        p = 0
        c = 0
        crazy_finder = []
        api = 0
        while True:
            _, frame = cap.read()
            width , height , z_pos = frame.shape
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(framergb)
            hand_landmarks = result.multi_hand_landmarks
            frame_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
            res = self.face_mesh2.process(frame_gray)
            dd1 = cv2.resize(frame,(200,200))
            my_list = []
            m_list = []
            if hand_landmarks:
                for handLMs in hand_landmarks:
                    for id , lm in enumerate(handLMs.landmark) :
                        if show_resulte != True :
                            time.sleep(0.1)
                        w , h , c1 = dd1.shape
                        x , y = int(lm.x*w) , int(lm.y*h)
                        my_list.append([id,x,y])
                        if len(my_list) > 20 :
                            x , y = my_list[8][1] , my_list[8][2]
                            x2 , y2 = my_list[6][1] , my_list[6][2]
                            x2w , y2w = my_list[5][1] , my_list[5][2]
                            x3 , y3 = my_list[12][1] , my_list[12][2]
                            x4 , y4 = my_list[10][1] , my_list[10][2]
                            x5 , y5 = my_list[4][1] , my_list[4][2]
                            x6 , y6 = my_list[3][1] , my_list[3][2]
                            a , b = my_list[8][1] , my_list[8][2]
                            c , d = my_list[12][1] , my_list[12][2]
                            e , f = my_list[16][1] , my_list[16][2]
                            g , h = my_list[20][1] , my_list[20][2]
                            a2 , b2 = my_list[6][1] , my_list[6][2]
                            c2 , d2 = my_list[10][1] , my_list[10][2]
                            e2 , f2 = my_list[14][1] , my_list[14][2]
                            g2 , h2 = my_list[19][1] , my_list[19][2]
                            g22 , h22 = my_list[17][1] , my_list[17][2]
                            cv2.line(frame , (x , y) , (x2 , y2) , (0,0,255) , 5)
                            k_list = ['finger 3' , 'finger 4' , 'all finger' , 'finger 1' , 'finger 2' , 'Thump' , 'Fist' ]
                            if b < b2 and d < d2 and f < f2 and h < h2 :
                                crazy_finder.append('all finger')
                            elif y2 > y :
                                crazy_finder.append('finger 1')
                            elif y4 > y3 :
                                crazy_finder.append('finger 2')
                            elif y2 > y5 :
                                crazy_finder.append('Thump')
                            elif h22 > h :
                                crazy_finder.append('finger 4')
                            else :
                                crazy_finder.append('Fist')
                                crazy_finder.append('Fist')
                            if len(crazy_finder) > 10 :
                                for n in k_list :
                                    nop = crazy_finder.count(n) 
                                    if nop > 8:
                                        api = n
                                        print(str(api))
                                crazy_finder = []
                            cv2.putText(frame , str(api) , (x5,y5) , cv2.FONT_HERSHEY_COMPLEX , 1 , (0,0,0) , 3)
                            self.mp_drawing.draw_landmarks(frame, handLMs, self.mphands.HAND_CONNECTIONS)
            if api == 'all finger' :
                notification.notify(title = 'time for take photo' ,message = f'picture saved with name ' , app_icon = r"C:\Users\smir1\Downloads\desktop (1) - Copy.ico" , timeout = 1)
                h = f'{file_name}.png'
                cv2.imwrite(r'C:\Users\smir1\OneDrive\Desktop\allfolder\computer_project\imagess\{}'.format(str(h)) , frame)
                break
            frame2 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            if show_resulte == True :
                cv2.imshow('hello',frame)
            cv2.waitKey(1)

n = ''
sfr = SimpleFacerec()
sfr.load_encoding_images(r"C:\Users\smir1\OneDrive\Desktop\allfolder\computer_project\imagess")
def start_find_contact(frame) :
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        n = name
        return n , x1 , y1 , x2 , y2


mphands = mp.solutions.hands
hands = mphands.Hands(12)
mp_drawing = mp.solutions.drawing_utils
def hand_tracking(frame , return_resulte = True , landmarks_id = 8 ) :
    my_list = []
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    hand_landmarks = result.multi_hand_landmarks

    if hand_landmarks:
        for handLMs in hand_landmarks:
            for id , lm in enumerate(handLMs.landmark) :
                w , h , c1 = framergb.shape
                x , y = int(lm.x*h) , int(lm.y*w)
                my_list.append([id , x , y])
                if len(my_list) > 20 : 
                    x1 , x2 = my_list[landmarks_id][1] , my_list[landmarks_id][2]
                    return x1 , x2
            mp_drawing.draw_landmarks(frame, handLMs, mphands.HAND_CONNECTIONS)





net = cv2.dnn.readNet(r"C:\Users\smir1\OneDrive\Desktop\yolov3.weights" , r"C:\Users\smir1\OneDrive\Desktop\yolov3.cfg")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def object_tracking(img) :
    height, width, channels = img.shape
    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), False, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:

            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

    return label , color , x , y , w , h

cap = cv2.VideoCapture(0)

face_mech = mp.solutions.face_mesh
face_mesh2 = face_mech.FaceMesh()
mppose = mp.solutions.pose
pose = mppose.Pose()
def face_tracking(frame , face_landmarks = 0) :
    try :
        frame_gray = cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
        res = face_mesh2.process(frame_gray)
        result34 = pose.process(frame_gray)
        pose_landmarks34 = result34.pose_landmarks
        for fl in res.multi_face_landmarks :
            face = fl.landmark[face_landmarks]
            h , w , z = frame.shape
            xp = int(face.x * w)
            yp = int(face.y * h)
            return xp , yp
    except : 
        None



try : 
    pos_x = 0
    pos_y = 0
    b = Arduino('COM3')
    b.digital[6].mode = SERVO
    b.digital[7].mode = SERVO
    def camera_movement(x1 , y1 , speed = 1) :
        global pos_x  
        global pos_y
        if x1 > 320 :
            pos_x = pos_x - speed
        if x1 < 320 :
            pos_x = pos_x + speed
        if y1 > 250 :
            pos_y = pos_y + speed
        if y1 < 250 :
            pos_y = pos_y - speed
        #print(pos)
        if (pos_x > 0 and pos_x < 255 ) : 
            b.digital[6].write(pos_x)
        if (pos_y > 0 and pos_y < 255 ) :
            b.digital[7].write(pos_y) 
        if pos_y > 255 : 
            pos_y = 0
        if pos_x > 255 : 
            pos_x = 0
        if pos_y < 0 : 
            pos_y = 0
        if pos_x < 0 : 
            pos_x = 0
except :
    None
