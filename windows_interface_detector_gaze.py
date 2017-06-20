#!/usr/bin/env python


import rospy


import rospy

import socket

import pymongo
from pymongo import MongoClient
import pprint
from bson.objectid import ObjectId

import argparse
import glob
import os

from scipy.spatial import distance

import sys

import sklearn

from sklearn import svm
import ast

import warnings
from OpenFace.msg import intent_msg
from OpenFace.msg import intent_msg_all


warnings.filterwarnings("ignore", category=DeprecationWarning)


continuos_intent = [0]*10
index_intent=0



class Pose_detection:
    def __init__(self, pose_tra_x, pose_tra_y, pose_tra_z, pose_rot_x, pose_rot_y, pose_rot_z, gaze_0_x, gaze_0_y,
                 gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, id_p):
        self.ptx = pose_tra_x
        self.pty = pose_tra_y
        self.ptz = pose_tra_z

        self.prx = pose_rot_x
        self.pry = pose_rot_y
        self.prz = pose_rot_z

        self.g0x = gaze_0_x
        self.g0y = gaze_0_y
        self.g0z = gaze_0_z

        self.g1x = gaze_1_x
        self.g1y = gaze_1_y
        self.g1z = gaze_1_z

        self.id = id_p

        self.looking = 0

    def looking_camera(self, clf):
        # using svm to do the classification
        vect = [self.prx, self.pry, self.prz]
        pred = clf.predict(vect)
        self.looking = pred
        return pred


class gesture_detection:
    def __init__(self, person_id, Head_pose_x, Head_pose_y, Head_pose_z, gesture_done):
        self.id = person_id
        self.head_pose_x = Head_pose_x
        self.head_pose_y = Head_pose_y
        self.head_pose_z = Head_pose_z
        self.gesture = gesture_done


class interaction_detect:
    def __init__(self, max_element_pose, max_element_gest, matrix_Transform_from_one_ref_to_other):

        self.dict_head_posex = {}
        self.dict_head_posey = {}
        self.dict_head_posez = {}
        self.dict_interaction = {}
        self.dict_looking = {}
        self.dict_gesture = {}
        self.person_id_list = []
        self.matrix = matrix_Transform_from_one_ref_to_other
        self.number_pose_Detected = max_element_pose
        self.number_gesture_Detected = max_element_gest
        self.number_of_persons = 0


    def matching(self,vector_pose_detections,vector_gesture_detection,matrix):


        # for each element in vector_pose_detection find the closest element and atribute a value to each connection (distance - norm 2)
        # optimization problem, find the closest point for each
        # start the atribution until there is no more atributions to do, if there is a element without atribution it means that element does not want to interact, but still have a label

        # Calculate for each one the correlection with the others , and do the matching. if distance between them is bigger than a thresold, decide that they are different persons
        # use the number of pose persons detected

        vector_norm_2_pose_gesture = []

        list_of_dict_value = []
        dict_of_vectors = {}

        print("NUMBER POSE DETECTED", self.number_pose_Detected)

        print("NUMBER gestures DETECTED", self.number_gesture_Detected)

        try:
            for i in range(self.number_pose_Detected):

                # Here we create the person and save if it looking to robot or not

                self.person_id_list.append(i)

                aux_norm_2 = []

                # Create a dictionary for values

                dict_for_values = {}

                # print("gaze coordinates")
                # print(vector_pose_detections[i].ptx)
                # print(vector_pose_detections[i].pty)
                # print(vector_pose_detections[i].ptz)

                # the distance between that person 'i' and model's person for gesture 'j'
                for j in range(self.number_gesture_Detected):
                    # calculate the distance
                    kinect_point = (vector_gesture_detection[j].head_pose_x, vector_gesture_detection[j].head_pose_y,
                                    vector_gesture_detection[j].head_pose_z)
                    gaze_point = (vector_pose_detections[i].ptx, vector_pose_detections[i].pty, vector_pose_detections[i].ptz)
                    distance_cal = distance.euclidean(kinect_point, gaze_point)

                    # print("kinect coordinates")
                    # print(vector_gesture_detection[j].head_pose_x)
                    # print(vector_gesture_detection[j].head_pose_y)
                    # print(vector_gesture_detection[j].head_pose_z)


                    # print("the distance calculated is", distance_cal)


                    # Dict to save the index of each distance
                    dict_for_values[distance_cal] = j

                    aux_norm_2.append(distance_cal)

                # Dict to save the index of each distance
                list_of_dict_value.append(dict_for_values)

                # Dict to save the index of each vector of distance
                dict_of_vectors[tuple(aux_norm_2)] = i
                vector_norm_2_pose_gesture.append(aux_norm_2)

            # Now that i have the distances between them, i have to do the matching

            # Start atributing first the small distance

            aux_vector_norm_2_pose_gesture = vector_norm_2_pose_gesture[:]

            # print ("VECOTRRRRRRR vector norm range!!!!!!!!!!",aux_vector_norm_2_pose_gesture)

            for k in range(len(vector_norm_2_pose_gesture)):

                # Find first the min vector of all of them
                min_vector = min(aux_vector_norm_2_pose_gesture)

                # GET THE REAL INDEX OF THAT VECTOR(POSE INDEX)  FROM DICT OF VECTORS
                index_original_vector_gaze = dict_of_vectors[tuple(min_vector)]

                # Get the index of min vector in the actual non atributed
                index_min_vector = aux_vector_norm_2_pose_gesture.index(min_vector)

                # get min value of min vector to do the matching in the non atributed
                min_value = min(min_vector)

                # GET THE REAL INDEX OF THAT VALUE(GSTURE INDEX)  FROM DICT OF VALUES
                index_original_value_gesture = list_of_dict_value[index_original_vector_gaze][min_value]

                # Do the matching only if the distance the euclidian distance is less than 2 meters
                # to prevent a case where there are 2 persons vailabel one looking and the other doing a gesture

                thresould_distance = 100.7
                # if (min_value < thresould_distance ):

                self.dict_gesture[index_original_vector_gaze] = vector_gesture_detection[index_original_value_gesture].gesture

                # Atribute values for pose of that person
                self.dict_head_posex[index_original_vector_gaze] = vector_gesture_detection[
                    index_original_value_gesture].head_pose_x
                self.dict_head_posey[index_original_vector_gaze] = vector_gesture_detection[
                    index_original_value_gesture].head_pose_y
                self.dict_head_posez[index_original_vector_gaze] = vector_gesture_detection[
                    index_original_value_gesture].head_pose_z
                self.number_of_persons += 1

                self.dict_looking[index_original_vector_gaze] = vector_pose_detections[index_original_vector_gaze].looking

                # Pop the assign vector
                new_vector = aux_vector_norm_2_pose_gesture.pop(index_min_vector)

                # del the value in dict and update the dict
                # dict_of_vectors[new_vector] = dict_of_vectors[min_vector]
                # del dict_of_vectors[min_vector]



                # Pop the elements from the list with the index atributed and update the dict of vectors
                for p in range(len(aux_vector_norm_2_pose_gesture)):

                    index_of_vector_to_update = dict_of_vectors[tuple(aux_vector_norm_2_pose_gesture[p])]
                    # UPDATE AUX VECTOR NORM 2 BY removing the element corresponding to the chosen index


                    for value in aux_vector_norm_2_pose_gesture[p]:
                        if (list_of_dict_value[index_of_vector_to_update][value] == index_original_value_gesture):
                            # AUX
                            aux_vector_bef_pop = aux_vector_norm_2_pose_gesture[p][:]

                            # THEN remove It FROM the list
                            aux_vector_norm_2_pose_gesture[p].remove(value)

                            # Update the dict
                            dict_of_vectors[tuple(aux_vector_norm_2_pose_gesture[p])] = dict_of_vectors[
                                tuple(aux_vector_bef_pop)]
                            del dict_of_vectors[tuple(aux_vector_bef_pop)]

            print("THIS IS THE FINAL MATCHING HERE!!!!!!!! for x", self.dict_head_posex)

        except:
            pass
        # print "No persons being detected!"

    def print_interactions(self,pub):

        global index_intent


        msg_all_interactions_results= intent_msg_all()
        msg_all_interactions_results.total_models = self.number_of_persons


        print("this are the interactions results")
        for i in range(self.number_of_persons):



            msg_interaction = intent_msg()
            msg_interaction.pose_tra_x = self.dict_head_posex[i]
            msg_interaction.pose_tra_y=  self.dict_head_posey[i]
            msg_interaction.pose_tra_z=  self.dict_head_posez[i]
            msg_interaction.looking =    self.dict_looking[i]
            msg_interaction.gesture =    self.dict_gesture[i]
            msg_interaction.result_interact = (self.dict_gesture[i]!=0 and self.dict_looking[i]==1)
            msg_interaction.box_h=0
            msg_interaction.box_w=0
            msg_interaction.box_x=0
            msg_interaction.box_y=0
            msg_interaction.id_model =i


            msg_all_interactions_results.intent_person.append(msg_interaction);



            print("Person with id:")
            print(i)
            print("At the pose :")
            print("x: ", self.dict_head_posex[i])
            print("y: ", self.dict_head_posey[i])
            print("z: ", self.dict_head_posez[i])

            print("Is doing the gesture:")
            print(self.dict_gesture[i])

            print("The result of looking is:")
            print( self.dict_looking[i])

            print("THE RESULT OF INTERACTION INTENT DETECTION IS")
            '''
            if(self.dict_gesture[i]!=0 and self.dict_looking[i]==1 ):
                print("YES!!!!!!!!!!!")
                continuos_intent[index_intent]=1
            else:
                print("NO")
                continuos_intent[index_intent] = 0
            if(index_intent==9):
                index_intent=0
            else:
                index_intent+=1

            if(continuos_intent.count(1)>continuos_intent.count(0)):
                print("PERSON",i, "WANTS TO INTERACT WITH GESTURE",self.dict_gesture[i])
            else:
                print("person",i,"does not want to interact :( ")
            '''
        pub.publish(msg_all_interactions_results)

class data_values:
    def __init__(self,id,tx,ty,tz,rx,ry,rz,g0x,g0y,g0z,g1x,g1y,g1z):

        self.pose_tra_x=float(tx)
        self.pose_tra_y=float(ty)
        self.pose_tra_z=float(tz)
        self.pose_rot_x=float(rx)
        self.pose_rot_y=float(ry)
        self.pose_rot_z=float(rz)
        self.gaze_0_rot_x=float(g0x)
        self.gaze_0_rot_y=float(g0y)
        self.gaze_0_rot_z=float(g0z)
        self.gaze_1_rot_x=float(g1x)
        self.gaze_1_rot_y=float(g1y)
        self.gaze_1_rot_z=float(g1z)
        self.id_model=float(id)

class data_process(data_values):

    def __init__(self, data_recev):
        #split by \n
        data_split = data_recev.split("\n")

        self.person = []
        self.total_models =int(float(data_split[-1]))

        for data_i in data_split[:-1]:
            #now we need to split data_i to create a new object
            #create new object inside
            ds = data_i.split(",")
            person_i = data_values(ds[0],ds[1],ds[2],ds[3],ds[4],ds[5],ds[6],ds[7],ds[8],ds[9],ds[10],ds[11],ds[12])
            self.person.append(person_i)


def gaze_callback(data_recv, args):
    db = args[0]
    clf = args[1]
    pub= args[2]


    data= data_process(data_recv)



    #as we are receiving from socket it is necessary to process information first

    # Every time a it is received a pose of someone, it is necessary to check if that person is looking to the camera and if it
    #  doing a gesture or not
    # persons detected by gaze detector
    persons_gaze_pose_det = []

    # here it is saved all information from gaze and pose detector
    for i in range(data.total_models):
        # creating a new person
        new_person = Pose_detection(data.person[i].pose_tra_x, data.person[i].pose_tra_y, data.person[i].pose_tra_z,
                                    data.person[i].pose_rot_x, data.person[i].pose_rot_y, data.person[i].pose_rot_z,
                                    data.person[i].gaze_0_rot_x, data.person[i].gaze_0_rot_y,
                                    data.person[i].gaze_0_rot_z, data.person[i].gaze_1_rot_x,
                                    data.person[i].gaze_1_rot_y, data.person[i].gaze_1_rot_z, data.person[i].id_model)
        # Check using the information taken from the gaze and pose detector who is looking to camera
        new_person.looking_camera(clf)
        # append to persons detected
        persons_gaze_pose_det.append(new_person)

    # Get the position of all the 6 persons who possible are doing a gesture, and save each one as an object of gestures detection

    valid_models = 0;
    # Get from database how many models are active and work only on those
    for valid_number in db.number_valid.find().sort("_id", pymongo.DESCENDING).limit(1):
        valid_models = valid_number['valid_number']
        print("number of valid models")
        print(valid_models)

    # Persobs detected by gesture detector
    persons_gesture_det = []
    for i in range(valid_models):
        # get the last frame for each person (THIS ONLY WORKS IN REAL TIME , NEED TO ADAPT TO DO EXPERIMENTS WITH PERSONS)
        for cursor in db.Gesture_info.find({'PersonID': i}).sort("_id", pymongo.DESCENDING).limit(1):
            # pprint.pprint(cursor)

            data_dict = cursor
            new_person_gest = gesture_detection(cursor['PersonID'], cursor['Head_pose_x'], cursor['Head_pose_y'],
                                                cursor['Head_pose_z'], cursor['gesture_done'])
            persons_gesture_det.append(new_person_gest)

    # Now it is available who did the gesture and who is looking to camera, this will be used to detect if the person wants to interact

    # need to be perform,instead of manually adjust
    matrix = 0

    # So now create an interaction detector object

    interaction_detector = interaction_detect(data.total_models, valid_models, matrix)

    # Adjust pose before send it to matching
    # kinect z axis points in the same direction of camera
    # gaze camera z axis points in the same direction of camera

    # kinect x axis points to the right of camera (perspective of person looking to camera)
    # gaze camera x axis points to the left of camera (perspective of person looking to camera)

    # kinect y axis points up (perspective of person looking to camera)
    # gaze camera y axis points down (perspective of person looking to camera)


    # First adjust the units, kinect in m, gaze is in mm
    # milimiters to meters

    # units
    for i in range(len(persons_gaze_pose_det)):
        persons_gaze_pose_det[i].ptx = persons_gaze_pose_det[i].ptx / 1000
        persons_gaze_pose_det[i].pty = persons_gaze_pose_det[i].pty / 1000
        persons_gaze_pose_det[i].ptz = persons_gaze_pose_det[i].ptz / 1000

    # there is need to adjust translaction, in KINECT REFERENTIAL
    # No need for z
    # lets admit they are at same level( no need to adjust y)
    # adjust the difference (if kinect is on the right and gaze on the left then)
    # the distance between them is negative an for example -0.5 meters

    # Translaction
    # x_distance_between_kinect_and_Gaze camera  in kinect referential
    x_dist = 0
    y_dist = 0
    z_dist = 0
    for i in range(len(persons_gaze_pose_det)):
        persons_gaze_pose_det[i].ptx = persons_gaze_pose_det[i].ptx + x_dist
        persons_gaze_pose_det[i].pty = persons_gaze_pose_det[i].pty + y_dist
        persons_gaze_pose_det[i].ptz = persons_gaze_pose_det[i].ptz + z_dist

    # ROTATION
    for i in range(len(persons_gaze_pose_det)):
        persons_gaze_pose_det[i].ptx = -persons_gaze_pose_det[i].ptx
        persons_gaze_pose_det[i].pty = -persons_gaze_pose_det[i].pty
        persons_gaze_pose_det[i].ptz = persons_gaze_pose_det[i].ptz

    # match the person with the gesture detected
    interaction_detector.matching(persons_gaze_pose_det, persons_gesture_det, matrix)

    print("start printing")
    interaction_detector.print_interactions(pub)
    '''
    try:
        interaction_detector.print_interactions()
    except:
        print("No persons being detected!")
    '''



        # classify if the person wants to interact or not


def listener(db, clf):


    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    # rospy.init_node('intention_detector', anonymous=True)
    # rospy.Subscriber('pose_gaze',pose_message_all , gaze_callback,(db,clf))
    # spin() simply keeps python from exiting until this node is stopped
    # rospy.spin()

    host = "10.0.29.88"
    port = 8889

    mySocket = socket.socket()
    mySocket.bind((host, port))

    mySocket.listen(1)
    conn, addr = mySocket.accept()
    print("Connection from: " + str(addr))

    rospy.init_node('intention_detector_win', anonymous=True)
    pub = rospy.Publisher('results_interaction_intent_win', intent_msg_all, queue_size=10)

    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print("from connected  user: " + str(data))
        args=[]
        args.append(db)
        args.append(clf)
        args.append(pub)
        #caling the callback  function
        gaze_callback(data, args)

        #data = str(data).upper()
        #print("sending: " + str(data))
        #conn.send(data.encode())

    conn.close()


def main():

    print(sys.argv)
    if len(sys.argv) < 2:
        print("insert ip adress as argv 0 , insert option camera as argv 1")
        return 0
    else:
        ip = sys.argv[1]
        print("ip adress selected as", ip)

    if len(sys.argv) == 3:
        option_camera = sys.argv[2]
        print("option camera selected as", option_camera)
    else:
        option_camera = 0
        print("default option camera selected as", option_camera)

    if len(sys.argv) > 3:
        print("insert ip adress as argv 0 , insert option camera as argv 1")
        return 0

    points = []
    label = []
    path = r'/home/jorgematos/image_transport_ws/training_looking_files'

    print("training model")
    for filename in glob.glob(os.path.join(path, '*.txt')):

        file = open(filename, "r")
        head, tail = os.path.split(filename)
        print(tail)

        if (tail[0] == 'n'):
            for line in file:
                line_list = ast.literal_eval(line)
                points.append(line_list)
                label.append(0)
        if (tail[0] == 'l'):
            for line in file:
                line_list = ast.literal_eval(line)
                points.append(line_list)
                label.append(1)

    clf = svm.SVC()
    clf.fit(points, label)

    print("model trained")

    client = MongoClient(ip, 27017)
    db = client.gestures_done

    vect= [0.03,0.334,0.56]
    pred = clf.predict(vect)

    print("this is preditction",pred)

    listener(db,clf)

if __name__ == "__main__":
    main()













#print("oi")