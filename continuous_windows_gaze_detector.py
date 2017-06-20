#!/usr/bin/env python


import rospy


from OpenFace.msg import intent_msg
from OpenFace.msg import intent_msg_all
from collections import Counter

import sys
import roslib
import sys
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
from cv_bridge import CvBridge, CvBridgeError
import glob
import copy
import ast
import os.path

from  scipy  import misc

from scipy.spatial import distance
from munkres import Munkres, print_matrix


LIFEFRAME=20

X_HEI= 1080  #480
Y_WID= 1920      #640


class person_intent:
 def __init__(self,px,py,pz,l,g,ri,bh,bw,bx,by,id):

    self.pose_tra_x =px
    self.pose_tra_y =py
    self.pose_tra_z =pz
    self.looking =l
    self.gesture =g
    self.result_interact =ri
    self.box_h =bh
    self.box_w =bw
    self.box_x =bx
    self.box_y =by
    self.id_model =id



class person_intent_cont(person_intent):
 def __init__(self,P):

    person_intent.__init__(self,P.pose_tra_x,P.pose_tra_y,P.pose_tra_z,P.looking,P.gesture,P.result_interact,P.box_h,P.box_w,P.box_x,P.box_y,P.id_model)
    self.index_max_update=30
    self.result_interact_cont=[0]* self.index_max_update
    self.result_of_interaction_med = P.result_interact
    self.index_update=0


    self.index_max_update_looking=30
    self.result_looking_cont=[0]* self.index_max_update
    self.result_of_looking_med = P.looking
    self.index_update_looking=0


    self.lifeFrame = LIFEFRAME


 def update(self,new_person):


    self.pose_tra_x =new_person.pose_tra_x
    self.pose_tra_ =new_person.pose_tra_y
    self.pose_tra_z =new_person.pose_tra_z
    self.looking =new_person.looking
    self.gesture = new_person.gesture
    self.result_interact = new_person.result_interact
    self.box_h = new_person.box_h
    self.box_w = new_person.box_w
    self.box_x = new_person.box_x
    self.box_y =new_person.box_y
    self.id_model =new_person.id_model


    #interaction intent
    self.result_interact_cont[self.index_update] = new_person.result_interact*new_person.gesture
    if self.index_update == self.index_max_update-1:
        self.index_update=0
    else:
        self.index_update+=1
    most_common,num_most_common = Counter(self.result_interact_cont).most_common(1)[0]
    self.result_of_interaction_med = most_common

    #looking
    self.result_looking_cont[self.index_update_looking] = new_person.looking
    if self.index_update_looking == self.index_max_update_looking-1:
        self.index_update_looking=0
    else:
        self.index_update_looking+=1
    most_common,num_most_common = Counter(self.result_looking_cont).most_common(1)[0]
    self.result_of_looking_med  = most_common






class listener_intent:

  def __init__(self):

    self.bridge = CvBridge()
    self.intent_sub = rospy.Subscriber("results_interaction_intent_win",intent_msg_all,self.callback_intent)


    self.intent_results = []
    self.intent_results_discrete = []

    self.total_persons_discrete = 0
    self.total_persons=0

    rospy.init_node('continuos_intention_detector_win', anonymous=True)
    self.pub = rospy.Publisher('continuos_intent_detector_win', intent_msg_all, queue_size=10)






  def callback_intent(self,data):
    self.total_persons_discrete=data.total_models
    self.intent_results_discrete=[]

    for i in range(data.total_models):
        #print(data.intent_person[i])
        new_person= person_intent(data.intent_person[i].pose_tra_x,data.intent_person[i].pose_tra_y,data.intent_person[i].pose_tra_z,data.intent_person[i].looking,data.intent_person[i].gesture,data.intent_person[i].result_interact,int(data.intent_person[i].box_h),int(data.intent_person[i].box_w),int(data.intent_person[i].box_x),int(data.intent_person[i].box_y),data.intent_person[i].id_model)
        self.intent_results_discrete.append(new_person)

    self.intent_continuos_update()

  def intent_continuos_update(self):

    #this represents the actual frame
    #self.intent_results_discrete

    #this represent the last frame
    #self.intent_results


    #first we try to do the matching using a distance not above 0.5 m , if fail create new persons or decrease lifetime frame

    #first calculate the distance between them and construct the matrix

    #row corresponds to new persons
    #column corresponds to saved persons

    #first case
    if len(self.intent_results)==0:
        for i in range(len(self.intent_results_discrete)):
            self.intent_results.append(person_intent_cont(self.intent_results_discrete[i]))
    else:
        matrix=[]
        for i in range(len(self.intent_results_discrete)):
            new_row_matrix=[]
            point_new = (self.intent_results_discrete[i].pose_tra_x,self.intent_results_discrete[i].pose_tra_y,self.intent_results_discrete[i].pose_tra_z)

            for j in range(len(self.intent_results)):
              #calculate the distance

                point_detected = (self.intent_results[j].pose_tra_x,self.intent_results[j].pose_tra_y,self.intent_results[j].pose_tra_z)

                distance_cal = distance.euclidean(point_detected,point_new)
                new_row_matrix.append(distance_cal)
            matrix.append(new_row_matrix)

        #print_matrix(matrix, msg=' matrix before assign:')

        if(len(matrix)>0):
            m = Munkres()
            indexes = m.compute(matrix)
            #print_matrix(matrix, msg='Lowest cost through this matrix:')
            #row means detect person , column means last detected person

            #this solves the case where the number of persons is equal to last frame
            for row, column in indexes:
                value = matrix[row][column]

                #print '(%d, %d) -> %f' % (row, column, value)
                aux=self.intent_results[column].lifeFrame
                self.intent_results[column].update(self.intent_results_discrete[row])

                if(self.intent_results[column].lifeFrame < LIFEFRAME):
                    self.intent_results[column].lifeFrame+=1


            #this solves the case where the number of persons detected is more than the last - adding person
            if len(indexes) < len(self.intent_results_discrete):

                print ("adding new person")

                #find the index not assign
                index_not_Assign_new_person = range(len(self.intent_results_discrete))
                for row, column in indexes:
                    index_not_Assign_new_person = list(filter(lambda x: x != row, index_not_Assign_new_person))
                for i in index_not_Assign_new_person:
                    self.intent_results.append(person_intent_cont(self.intent_results_discrete[i]))



            #this solves the case where the number of persons detected is less than the last - decrease live points
            if len(indexes) < len(self.intent_results):
                #remove one life point
                #find the index not assign

                print("someone dissapear")

                index_not_Assign_new_person = range(len(self.intent_results))
                for row, column in indexes:
                    index_not_Assign_new_person = list(filter(lambda x: x != column, index_not_Assign_new_person))
                for i in index_not_Assign_new_person:
                    self.intent_results[i].lifeFrame-=1
                #if have 0 life points then die
                for i in range(len(self.intent_results)):
                    if(self.intent_results[i].lifeFrame==0):
                        #die
                        self.intent_results.pop(i)
        else:
        #if no one was detected kill every one a point
            for i in range(len(self.intent_results)):
                    self.intent_results[i].lifeFrame-=1
                #if have 0 life points then die
            for i in range(len(self.intent_results)):
                if(self.intent_results[i].lifeFrame==0):
                    #die
                    self.intent_results.pop(i)


    #publish results here!!!!!!!
    msg_all_interactions_results= intent_msg_all()
    msg_all_interactions_results.total_models = len(self.intent_results)


    for i in range(len(self.intent_results)):



        msg_interaction = intent_msg()
        msg_interaction.pose_tra_x = self.intent_results[i].pose_tra_x
        msg_interaction.pose_tra_y=  self.intent_results[i].pose_tra_y
        msg_interaction.pose_tra_z=  self.intent_results[i].pose_tra_z
        msg_interaction.looking =    self.intent_results[i].result_of_looking_med
        msg_interaction.gesture =    self.intent_results[i].gesture
        msg_interaction.result_interact =   self.intent_results[i].result_of_interaction_med
        msg_interaction.box_h=0
        msg_interaction.box_w=0
        msg_interaction.box_x=0
        msg_interaction.box_y=0
        msg_interaction.id_model =i
        msg_all_interactions_results.intent_person.append(msg_interaction);

    pub.publish(msg_all_interactions_results)




def main(args):
  ic = listener_intent()
  rospy.init_node('intent_listener', anonymous=True)


  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()



if __name__ == '__main__':


    #image = [cv2.imread("box.png") in glob.glob("/home/jorgematos/image_transport_ws/images")]
    '''
    img_box=  cv2.imread('images/vislab_eye_yes.png')
    cv2.imshow("Image windowl", img_box)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    '''


    main(sys.argv)