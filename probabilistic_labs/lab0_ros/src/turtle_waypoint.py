#!/usr/bin/env python
import roslib; roslib.load_manifest('lab0_ros')
import rospy

#For command line arguments
import sys
#For atan2
import math

#TODO: Import the messages we need
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist
##

#Initialization of turtle position
x=0.0
y=0.0
theta=0.0

#Position tolerance for both x and y
tolerance=0.1
#Have we received any pose msg yet?
gotPosition=False

def callback(pose_msg):
    global x,y,theta,gotPosition
    #TODO:Store the position in x,y and theta variables
    x=pose_msg.x
    y=pose_msg.y
    theta=pose_msg.theta
    gotPosition=True

def waypoint():
    global gotPosition
    #TODO: Define the pulisher: Name of the topic. Type of message
    pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=1)

    #Name of the node
    rospy.init_node('turtle_waypoint')
    #TODO: Define the subscriber: Name of the topic. Type of message. Callback function
    sub = rospy.Subscriber('/turtle1/pose', Pose, callback)

    #Has the turtle reach position?
    finished=False
    #If the point hasn't been specified in a command line:
    if(len(sys.argv)!=3):
        print('X and Y values not set or not passed correctly. Looking for default parameters.')
        #TODO: If ROS parameters default_x and default_y exist:
        if rospy.has_param('default_x') and rospy.has_param('default_y'): #Change this for the correct expression
            #TODO: Save them into this variables
            x_desired=rospy.get_param('default_x') #Change this for the correct expression
            y_desired=rospy.get_param('default_y') #Change this for the correct expression
            print('Heading to: %f,%f' %(x_desired, y_desired))
            print('Rotating to the perfect angle')
            gotPosition=True
        else:
            print('Default values parameters not set!. Not moving at all')
            finished=True
    else:
        #Save the command line arguments.
        x_desired=float(sys.argv[1])
        y_desired=float(sys.argv[2])
        print('Heading to: %f,%f' %(x_desired, y_desired))
        print('Rotating to the perfect angle')
        gotPosition=True

    tst =Twist()
    #Rotating in the appropriate direction
    while not rospy.is_shutdown() and not finished:
        if(gotPosition):
            #TODO: Send a velocity command for every loop until the position is reached within the tolerance.
            diffx=x_desired-x
            diffy=y_desired-y
            teta=math.atan2(diffy,diffx)
            if teta<0:
                teta=2*math.pi+teta
            if(math.fabs(teta - theta)<0.005):
                tst.angular.z=0    
                pub.publish(tst)
                print('Moving forward')
                break
            tst.angular.z=(teta-theta)  
            pub.publish(tst)
    #Moving forward towards the desired location
    while not rospy.is_shutdown() and not finished:
        if(gotPosition):
            diffx=x_desired-x
            diffy=y_desired-y
            if(math.fabs(diffx) >= tolerance) or  (math.fabs(diffy) >= tolerance):
                tst.linear.x = math.sqrt(math.pow(diffx,2) + math.pow(diffy,2))        
            else:
                print('Reached desired location')
                tst.linear.x = 0                   
                tst.angular.z = 0                     
                finished=True        

            pub.publish(tst)

        #Publish velocity commands every 0.3 sec.
        rospy.sleep(0.3)

if __name__ == '__main__':
    try:
        waypoint()
    except rospy.ROSInterruptException:
        pass
