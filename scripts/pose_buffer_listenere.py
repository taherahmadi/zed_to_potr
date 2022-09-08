#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String
from pose_publisher.msg import Skeleton3DBuffer

# buffer is a queue

def callback(data):    
    print("data:", np.array(data.skeleton_3d_17_flat.data).reshape(data.shape))

def listener():
    rospy.init_node('pose_publisher', anonymous=True)

    rospy.Subscriber('/pose_publisher/3DSkeletonBuffer', Skeleton3DBuffer, callback)
    # pub.publish(pose_buffer)
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
if __name__ == '__main__':
    listener()
