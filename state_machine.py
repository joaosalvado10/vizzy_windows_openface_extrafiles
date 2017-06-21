#!/usr/bin/env python

import rospy
import smach
import smach_ros

import threading

from OpenFace.msg import intent_msg
from OpenFace.msg import intent_msg_all

MAX_VALUE_GESTURE =300

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



# define state Stop
class Stop(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['No_one_detected', 'person_detected'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])

        self.mutex = threading.Lock()
        self.intent_sub = rospy.Subscriber("'continuos_intent_detector_win'",intent_msg_all,self.callback)
        self.id_chosen=-1




    def callback(self, data):

        #self.mutex.acquire()
        #chose the closest person
        distance=9999
        for i in range(data.total_models):
            #print(data.intent_person[i])
            new_person= person_intent(data.intent_person[i].pose_tra_x,data.intent_person[i].pose_tra_y,data.intent_person[i].pose_tra_z,data.intent_person[i].looking,data.intent_person[i].gesture,data.intent_person[i].result_interact,int(data.intent_person[i].box_h),int(data.intent_person[i].box_w),int(data.intent_person[i].box_x),int(data.intent_person[i].box_y),data.intent_person[i].id_model)
            if new_person.pose_tra_z < distance:
                distance=new_person.pose_tra_z
                self.id_chosen = new_person.id_model
        #self.mutex.release()





    def execute(self, userdata):
        rospy.loginfo('Executing state STOP')
        #means we have detected a person
        #self.mutex.acquire()
        if self.id_chosen != -1:
            userdata.id_person_detected = self.id_chosen
            self.id_chosen=-1
            return 'person_detected'
        #means no one was detecte yet
        else:
            return 'No_one_detected'
        #self.mutex.release()





# define state Follow
class Follow(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['following','lost_person_in_follow','reach_person'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])
        self.list_of_persons=[]
        self.mutex = threading.Lock()
        self.intent_sub = rospy.Subscriber("'continuos_intent_detector_win'",intent_msg_all,self.callback)



    def callback(self, data):

        #self.mutex.acquire()
        self.list_of_persons=[]
        for i in range(data.total_models):
            #print(data.intent_person[i])
            new_person= person_intent(data.intent_person[i].pose_tra_x,data.intent_person[i].pose_tra_y,data.intent_person[i].pose_tra_z,data.intent_person[i].looking,data.intent_person[i].gesture,data.intent_person[i].result_interact,int(data.intent_person[i].box_h),int(data.intent_person[i].box_w),int(data.intent_person[i].box_x),int(data.intent_person[i].box_y),data.intent_person[i].id_model)
            self.list_of_persons.append(new_person)
        #self.mutex.release()



    def execute(self, userdata):
        rospy.loginfo('Executing state FOLLOW')
        #first try to find person
        #self.mutex.acquire()

        lost =1
        for person in self.list_of_persons:
            #means we have person and we can follow the person or go to next stage if we are close enought
            if(person.id_model==userdata.id_person_detected):
                lost=0
                #if we are 1 meter far from vizzy -> approach more
                if(person.pose_tra_z >1):

                    #PUBLISH TO ROS TOPIC HERE!!!!!!!!!!!!!!!!!!!!
                    return 'following'

                #Means we already reach the person
                else:
                    return 'reach_person'
        #means we did not find the id passed by argument
        if(lost==1):
            return 'lost_person_in_follow'
        #self.mutex.release()


# define state Speack
class Speack(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['fail_speack','speacked'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])
        self.mutex = threading.Lock()

    def execute(self, userdata):
        rospy.loginfo('Executing state SPEACK')

        result_from_action_speack = ACTION_SPEACK
        if(result_from_action_speack==1):
            return 'speacked'
        else:
            return 'fail_speack'


# define state Do_gesture
class Do_gesture(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['fail_doing_gesture','succeed_doing_gesture'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])


    def execute(self, userdata):
        rospy.loginfo('Executing state DO_GESTURE')

        result_from_action_do_gesture = ACTION_GESTURE
        if(result_from_action_do_gesture==1):
            return 'succeed_doing_gesture'
        else:
            return 'fail_doing_gesture'

# define state Detect_gesture
class Detect_gesture(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['detecting_gesture','gesture_detected','fail_to_detect_gesture'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])

        self.list_of_persons=[]
        self.mutex = threading.Lock()
        self.intent_sub = rospy.Subscriber("'continuos_intent_detector_win'",intent_msg_all,self.callback)

        self.counter=0

    def callback(self, data):

        #self.mutex.acquire()
        self.list_of_persons=[]
        for i in range(data.total_models):
            #print(data.intent_person[i])
            new_person= person_intent(data.intent_person[i].pose_tra_x,data.intent_person[i].pose_tra_y,data.intent_person[i].pose_tra_z,data.intent_person[i].looking,data.intent_person[i].gesture,data.intent_person[i].result_interact,int(data.intent_person[i].box_h),int(data.intent_person[i].box_w),int(data.intent_person[i].box_x),int(data.intent_person[i].box_y),data.intent_person[i].id_model)
            self.list_of_persons.append(new_person)
        #self.mutex.release()



    def execute(self, userdata):
        global MAX_VALUE_GESTURE

        rospy.loginfo('Executing state DETECT_GESTURE')
        lost =1
        for person in self.list_of_persons:
            #means we are able to detect the gesture
            if(person.id_model==userdata.id_person_detected):
                lost=0
                #if we still havent reached the maximum time
                if(self.counter < MAX_VALUE_GESTURE):
                    #CHECK IF HAND WAVE IS 2!!!!
                    if( person.gesture==2 or person.gesture==3  ):
                        self.counter=0
                        return 'gesture_detected'
                    else:
                        self.counter+=1
                        return 'following'

                #maximum time reached
                else:
                    self.counter=0
                    return 'fail_to_detect_gesture'

        #means we did not find the id passed by argument
        if(lost==1):
            return 'fail_to_detect_gesture'
        #self.mutex.release()



# define state Detect_gesture
class Go_to_point(smach.State):
    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['fail_Reach_the_point','point_reached'],
                             input_keys=['id_person_detected'],
                             output_keys=['id_person_detected'])

    def execute(self, userdata):
        rospy.loginfo('Executing state GO_TO_POINT')


        userdata.id_person_detected=-1
        return 'outcome1'
        result_from_action_go_to_point = ACTION_GO_TO_POINT
        if(result_from_action_go_to_pointk==1):
            return 'point_reached'
        else:
            return 'fail_Reach_the_point'






# main
def main():

    rospy.init_node('smach_state_machine')
    rospy.spin()


    # Create a SMACH state machine
    sm = smach.StateMachine(outcomes=['No_one_detected', 'person_detected','following','lost_person_in_follow','reach_person','fail_speack','speacked','fail_doing_gesture',
        'succeed_doing_gesture','detecting_gesture','gesture_detected','fail_to_detect_gesture','fail_Reach_the_point','point_reached'])

    #variable about the person being detected
    sm.userdata.id_person_detected = -1


    # Open the container
    with sm:
        # Add states to the container
        smach.StateMachine.add('STOP', Stop(),
                               transitions={'No_one_detected':'STOP',
                                            'person_detected':'FOLLOW'},

                               remapping={'Stop_id_in':'id_person_detected',
                                        'Stop_id_out':'id_person_detected'})

        smach.StateMachine.add('FOLLOW', Follow(),
                               transitions={'following':'FOLLOW',
                                            'lost_person_in_follow':'STOP',
                                            'reach_person':'SPEACK'},
                               remapping={'Follow_id_in':'id_person_detected',
                                        'Follow_id_out':'id_person_detected'})

        smach.StateMachine.add('SPEACK', Speack(),
                               transitions={'fail_speack':'STOP',
                                            'speacked':'DO_GESTURE'},
                               remapping={'Speack_id_in':'id_person_detected',
                                        'Speack_id_out':'id_person_detected'})
        smach.StateMachine.add('DO_GESTURE', Do_gesture(),
                               transitions={'fail_doing_gesture':'STOP',
                                            'succeed_doing_gesture':'DETECT_GESTURE'},
                               remapping={'Do_gesture_id_in':'id_person_detected',
                                        'Do_gesture_id_out':'id_person_detected'})

        smach.StateMachine.add('DETECT_GESTURE', Detect_gesture(),
                               transitions={'detecting_gesture':'DETECT_GESTURE',
                                            'gesture_detected':'GO_TO_POINT',
                                            'fail_to_detect_gesture':'STOP'},
                               remapping={'Detect_gesture_id_in':'id_person_detected',
                                        'Detect_gesture_id_out':'id_person_detected'})
        smach.StateMachine.add('GO_TO_POINT', Go_to_point(),
                               transitions={'fail_Reach_the_point':'STOP',
                                            'point_reached':'STOP'},
                               remapping={'Go_point_id_in':'id_person_detected',
                                        'Go_point_id_out':'id_person_detected'})



    # Execute SMACH plan
    outcome = sm.execute()


if __name__ == '__main__':
    main()
