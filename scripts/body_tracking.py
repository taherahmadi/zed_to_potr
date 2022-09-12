########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
from audioop import avg
from turtle import pos
from unittest import skip
from xml.dom.pulldom import default_bufsize
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np
import rospy
from std_msgs.msg import String
from std_msgs.msg import Header
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import MultiArrayDimension
from pose_publisher.msg import Skeleton3D17
from pose_publisher.msg import Skeleton3DBuffer
from zed_interfaces.msg import Keypoint3D
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
import tf
import tf.transformations as tr
import time
import math



ADE_pose = []
FDE_pose = []
ADE_traj = []
FDE_traj = []


def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix


class zed_to_potr():

    def __init__(self):
        # Create a Camera object
        self.zed = sl.Camera()
        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD7 video mode
        self.init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
        self.init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
        # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOWN
        self.init_params.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP # for ROS coordinate system
        self.time_retrieve = time.time()
        self.time_frame_rate = time.time()

        # If applicable, use the SVO given as parameter
        # Otherwise use ZED live stream
        if len(sys.argv) == 2:
            filepath = sys.argv[1]
            print("Using SVO file: {0}".format(filepath))
            self.init_params.svo_real_time_mode = True
            self.init_params.set_from_svo_file(filepath)

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        # Enable Positional tracking (mandatory for object detection)
        self.positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
        # positional_tracking_parameters.set_as_static = True
        self.zed.enable_positional_tracking(self.positional_tracking_parameters)
        
        self.obj_param = sl.ObjectDetectionParameters()
        self.obj_param.enable_body_fitting = True            # Smooth skeleton move
        self.obj_param.enable_tracking = True                # Track people across images flow
        self.obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
        self.obj_param.body_format = sl.BODY_FORMAT.POSE_34  # Choose the BODY_FORMAT you wish to use

        # Enable Object Detection module
        self.zed.enable_object_detection(self.obj_param)

        self.skeleton_publisher = rospy.Publisher('/pose_publisher/3DSkeletonBuffer', Skeleton3DBuffer, queue_size=1)
        # rospy.Subscriber('/potrtr/predictions', Skeleton3DBuffer, self.predictions_eval) # no longer needed

        self.pc1_publisher = []
        for i in range(5):
            self.pc1_publisher.append(rospy.Publisher('/pose_publisher/skeleton'+str(i), PointCloud, queue_size=1))

        self.tf_br = tf.TransformBroadcaster()

        self.rate = 100
        self.r = rospy.Rate(self.rate)

        self.pose_buffer = []
        self.dt_buffer = []
        self.buffer_size = 20
        self.output_size = 5
        self.desired_framerate = 10
        self.transfrom_for_seq = {}


    def body_tracking(self):
        print("Running Body Tracking sample ... Press 'q' to quit")

        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 40

        # Get ZED camera information
        camera_info = self.zed.get_camera_information()

        # 2D viewer utilities
        display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
        image_scale = [display_resolution.width / camera_info.camera_resolution.width
                    , display_resolution.height / camera_info.camera_resolution.height]

        # Create OpenGL viewer
        # viewer = gl.GLViewer()
        # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

        # Create ZED objects filled in the main loop
        bodies = sl.Objects()
        # image = sl.Mat()
        
        runtime_params = sl.RuntimeParameters()
        runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

        time0 = time.time()
        # while viewer.is_available():
        i = 0
        while not rospy.is_shutdown():
            i +=1
            #start time
            
            # Grab an image
            if self.zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
                break
            else:

                # Retrieve left image
                # self.zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                # Retrieve objects
                self.zed.retrieve_objects(bodies, obj_runtime_param)

                # Get the pose of the camera relative to the world frame
                camera_pose = sl.Pose()
                py_translation = sl.Translation()
                state = self.zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD) #didn't work proly
                # camera odom
                camera_translation = camera_pose.get_translation(py_translation).get() 
                # print("Camera Translation: ", camera_translation)
                # orientation quaternion
                py_orientation = sl.Orientation()
                camera_orientation = camera_pose.get_orientation(py_orientation).get()

                self.tf_br.sendTransform( camera_translation, # bodies.object_list[0].keypoint[0], # translation
                                        camera_orientation, # bodies.object_list[0].global_root_orientation, # (x, y, z, w) rotation
                                        rospy.Time.now(),
                                        'camera',
                                        'world')
                
                
                if time.time() - self.time_retrieve < 0.09:
                    self.r.sleep()
                    continue
                self.time_retrieve = time.time()


                if len(bodies.object_list)>0:
                    body_index = 0
                    if len(bodies.object_list)>1 and len(self.pose_buffer)>1:
                        temp = self.pose_buffer[-1]
                        dif = []
                        for body in bodies.object_list:
                            dif.append(np.sum((np.array(temp[0])-np.array(body.keypoint[0]))**2, axis=0))
                        body_index = dif.index(min(dif))
                    C_to_W_T = self.publish_skeleton_tf(bodies, camera_translation, camera_orientation, body_index=body_index)
                    self.publish_17skeleton(bodies, C_to_W_T, body_index=body_index)

                # Update GL view
                # viewer.update_view(image, bodies) 
                # # Update OCV view
                # image_left_ocv = image.get_data() 
                # cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
                # cv2.imshow("ZED | 2D View", image_left_ocv)
                # cv2.waitKey(10)
                self.r.sleep()
                

        # viewer.exit()
        # image.free(sl.MEM.CPU)
        # Disable modules and close camera
        self.zed.disable_object_detection()
        self.zed.disable_positional_tracking()
        self.zed.close()

    def publish_17skeleton(self, bodies, C_to_W_T, body_index=0):

        W_to_C_T = np.linalg.inv(C_to_W_T) # Pose of frame World relative to frame Camera

        # change skeleton data here
        self.pose_buffer.append(zed32_to_17_format(bodies.object_list[body_index].keypoint))
        self.pose_buffer = self.pose_buffer[-25:]

        self.dt_buffer.append(time.time()-self.time_frame_rate)
        self.time_frame_rate = time.time()
        self.dt_buffer = self.dt_buffer[-25:]
        if len(self.pose_buffer)>=self.output_size:
            avg_framerate = 1/np.mean(self.dt_buffer)
            print("avg frame rate: ", avg_framerate)
            k = max(1,math.floor(avg_framerate/self.desired_framerate))

            for i in range(5):
                # TODO; this was wrong, we need to fix it for different frame rate
                # print(pose_buffer[-output_size+i].shape)
                # new_pose_in_camera = pose_transform(pose_buffer[-output_size+i].reshape(1, 17, 3), C_to_W_T)
                self.pc1_publisher[i].publish(get_point_clouds(np.array(self.pose_buffer[-self.output_size+i]), relative_to="world"))

            pose_buffer_msg = Skeleton3DBuffer()
            array_msg =  Float64MultiArray()
            # get_3dpose_in_camera_frame
            output_buffer = pose_transform(np.array(self.pose_buffer[-1*self.output_size:]), Transform=W_to_C_T)
            output_buffer_shape = output_buffer.shape
            array_msg.data = output_buffer.flatten()

            pose_buffer_msg.skeleton_3d_17_flat = array_msg
            pose_buffer_msg.skeleton_3d_17_flat_shape = output_buffer_shape

            transfrom_array_msg =  Float64MultiArray()
            transfrom_array_msg.data =  np.array(C_to_W_T).flatten()
            pose_buffer_msg.transform = transfrom_array_msg
            pose_buffer_msg.transform_shape = [4,4]
            self.skeleton_publisher.publish(pose_buffer_msg)


    def publish_skeleton_tf(self, bodies, camera_translation, camera_orientation, body_index=0):
        # Camera to World homogeneous transformation matrix
        C_to_W_T = np.eye(4) # Pose of frame Camera relative to frame w
        camera_rotation = quaternion_to_rotation_matrix(camera_orientation)
        C_to_W_T[:3,:3] = camera_rotation
        C_to_W_T[:3,3] = camera_translation
        
        # Get the pose of the Human relative to the world frame
        # translation
        human_translation = bodies.object_list[body_index].keypoint[body_index]
        # orientation quaternion
        human_orientation = bodies.object_list[body_index].global_root_orientation

        # Camera to World homogeneous transformation matrix
        H_to_W_T = np.eye(4)  # pose of the frame human relative to frame world
        human_rotation = quaternion_to_rotation_matrix(human_orientation)
        H_to_W_T[:3,:3] = human_rotation
        H_to_W_T[:3,3] = human_translation
        human_to_world_rotation = H_to_W_T[:3,:3]
        human_to_world_translation = H_to_W_T[:3,3]

        # H_to_C_T=np.eye(4) # pose of the frame human relative to frame camera
        # H_to_C_T = np.matmul(W_to_C_T, H_to_W_T)
        # human_to_camera_rotation = H_to_C_T[:3,:3]
        # human_to_camera_translation = H_to_C_T[:3,3]

        h2w_R = np.eye(4)
        h2w_R[:3,:3] = human_to_world_rotation
        self.tf_br.sendTransform( human_to_world_translation, # Human frame translation realtive to World
                        tr.quaternion_from_matrix(h2w_R), # bodies.object_list[0].global_root_orientation, # (x, y, z, w) rotation
                        rospy.Time.now(),
                        'skeleton_17_pc',
                        'world')
        
        return C_to_W_T
    
 

def get_point_clouds(skeleton_3d, relative_to="world"):
    pc = PointCloud()
    pc.header.stamp = rospy.Time.now()
    pc.header.frame_id = relative_to
    for i in range(skeleton_3d.shape[0]):
        pc.points.append(Point32(x=skeleton_3d[i][0],y=skeleton_3d[i][1],z=skeleton_3d[i][2]))
    return pc


def zed32_to_17_format(skeleton_3d):
    skeleton_3d_17 = np.zeros((17,3))
    if len(skeleton_3d) > 0:
        # POSE_34
        # body_format == sl.BODY_FORMAT.POSE_34
        if len(skeleton_3d) == 34:
            # skeleton_3d_17[0] = skeleton_3d[0]
            # skeleton_3d_17[1] = skeleton_3d[18]
            # skeleton_3d_17[2] = skeleton_3d[19]
            # skeleton_3d_17[3] = skeleton_3d[20]
            # skeleton_3d_17[4] = skeleton_3d[22]
            # skeleton_3d_17[5] = skeleton_3d[23]
            # skeleton_3d_17[6] = skeleton_3d[24]
            # skeleton_3d_17[7] = skeleton_3d[2]
            # skeleton_3d_17[8] = skeleton_3d[3]
            # skeleton_3d_17[9] = skeleton_3d[26]
            # skeleton_3d_17[10] = skeleton_3d[27]
            # skeleton_3d_17[11] = skeleton_3d[12]
            # skeleton_3d_17[12] = skeleton_3d[13]
            # skeleton_3d_17[13] = skeleton_3d[14]
            # skeleton_3d_17[14] = skeleton_3d[5]
            # skeleton_3d_17[15] = skeleton_3d[6]
            # skeleton_3d_17[16] = skeleton_3d[7]
            new_indices = [0,18,19,20,22,23,24,1,3,26,27,12,13,14,5,6,7]
            skeleton_3d_17 = skeleton_3d[new_indices]
            skeleton_3d_17[10] = (skeleton_3d[29]+skeleton_3d[31])/2
            skeleton_3d_17[8] = 0.75*skeleton_3d_17[8]+0.25*skeleton_3d[27]
    return skeleton_3d_17

def pose_transform(skeleton_buffer, Transform):
    skeleton_buffer = np.concatenate((skeleton_buffer, np.ones((skeleton_buffer.shape[0],skeleton_buffer.shape[1],1))),axis=2)
    skeleton_buffer_shape = skeleton_buffer.shape
    skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0]*skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
    skeleton_buffer_t = np.transpose(skeleton_buffer)
    skeleton_buffer_t = np.matmul(Transform, skeleton_buffer_t)
    skeleton_buffer = np.transpose(skeleton_buffer_t)
    skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0],skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
    return skeleton_buffer[:,:,:3]



if __name__ == "__main__":

    #ros node initialization 
    rospy.init_node('pose_publisher', anonymous=True)

    pose_publisher = zed_to_potr()
    pose_publisher.body_tracking()

    rospy.spin()

