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
from numpy_ros import to_numpy, to_message




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
def multiarray_to_numpy(pytype, dtype, multiarray):
    dims = tuple(map(lambda x: x.size, multiarray.layout.dim))
    return np.array(multiarray.data, dtype=pytype).reshape(dims).astype(dtype)

def pose3d_zed():
    print("Running Body Tracking sample ... Press 'q' to quit")


    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    # init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_DOWN
    init_params.coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP # for ROS coordinate system
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_ACCURATE
    obj_param.body_format = sl.BODY_FORMAT.POSE_34  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()
    
     # pose_data = sl.Transform()          
    text_translation = ""
    text_rotation = ""

    runtime_params = sl.RuntimeParameters()
    runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

    #ros node initialization 
    rospy.init_node('pose_publisher', anonymous=True)
    publisher = rospy.Publisher('/pose_publisher/3DSkeletonBuffer', Skeleton3DBuffer, queue_size=10)
    # pc1_publisher = rospy.Publisher('/pose_publisher/skeleton1', PointCloud, queue_size=10)
    # pc2_publisher = rospy.Publisher('/pose_publisher/skeleton2', PointCloud, queue_size=10)
    pose_buffer = []
    dt_buffer = []
    buffer_size = 20
    output_size = 5
    desired_framerate = 10

    time0 = time.time()
    # while viewer.is_available():
    while True:
        
        i = 0
        #start time
        time1 = time.time()
        
        # Grab an image
        if zed.grab(runtime_params) != sl.ERROR_CODE.SUCCESS:
            break
        else:

            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)
            
            if len(bodies.object_list)==1:

                # Get the pose of the camera relative to the world frame
                camera_pose = sl.Pose()
                py_translation = sl.Translation()
                state = zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                # translation
                camera_translation = camera_pose.get_translation(py_translation).get() 
                # print("Camera Translation: ", camera_translation)
                # orientation quaternion
                py_orientation = sl.Orientation()
                camera_orientation = camera_pose.get_orientation(py_orientation).get()
                # print("Camera Orientation: ", camera_orientation)

                # Camera to World homogeneous transformation matrix
                C_to_W_T = np.eye(4) # Pose of frame Camera relative to frame w
                camera_rotation = quaternion_to_rotation_matrix(camera_orientation)
                C_to_W_T[:3,:3] = camera_rotation
                C_to_W_T[:3,3] = camera_translation

                W_to_C_T = np.linalg.inv(C_to_W_T) # Pose of frame World relative to frame Camera
                
                # Get the pose of the Human relative to the world frame
                # translation
                human_translation = bodies.object_list[0].keypoint[0]
                # print("Human Translation: ",human_translation)
                # orientation quaternion
                human_orientation = bodies.object_list[0].global_root_orientation
                # print("Human Orientation: ", human_orientation)

                # Camera to World homogeneous transformation matrix
                H_to_W_T = np.eye(4)  # pose of the frame human relative to frame world
                human_rotation = quaternion_to_rotation_matrix(human_orientation)
                H_to_W_T[:3,:3] = human_rotation
                H_to_W_T[:3,3] = human_translation
                # human_to_world_rotation = H_to_W_T[:3,:3]
                # human_to_world_translation = H_to_W_T[:3,3]

                # H_to_C_T=np.eye(4) # pose of the frame human relative to frame camera
                # H_to_C_T = np.matmul(W_to_C_T, H_to_W_T)
                # human_to_camera_rotation = H_to_C_T[:3,:3]
                # human_to_camera_translation = H_to_C_T[:3,3]

                # br = tf.TransformBroadcaster()
                # br.sendTransform( camera_translation, # bodies.object_list[0].keypoint[0], # translation
                #                   camera_orientation, # bodies.object_list[0].global_root_orientation, # (x, y, z, w) rotation
                #                   rospy.Time.now(),
                #                   'camera',
                #                   'world')
                # # h2w_R = np.eye(4)
                # h2w_R[:3,:3] = human_to_world_rotation
                # br.sendTransform( human_to_world_translation, # Human frame translation realtive to World
                #                   tr.quaternion_from_matrix(h2w_R), # bodies.object_list[0].global_root_orientation, # (x, y, z, w) rotation
                #                   rospy.Time.now(),
                #                   'skeleton_17_pc',
                #                   'world')
            

                # change skeleton data here
                pose_buffer.append(zed32_to_17_format(bodies.object_list[0].keypoint))
                dt_buffer.append(time.time()-time1)
                # publish point clouds 
                # pc1_publisher.publish(get_point_clouds(bodies.object_list[0], W_to_C_T, br, relative_to="world"))
                # pc2_publisher.publish(get_point_clouds(bodies.object_list[0], W_to_C_T, br, relative_to="camera"))


            if len(pose_buffer)>buffer_size:
                pose_buffer.pop(0)
                dt_buffer.pop(0)

            # if elapsed time is 0.5 sec
            if (time.time()- time0) >= 0.5:
                time0 = time.time()


                if len(pose_buffer)>=output_size:
                    avg_framerate = 1/np.mean(dt_buffer)
                    print("avg frame rate: ", avg_framerate)
                    k = max(1,math.floor(avg_framerate/desired_framerate))
                
                    pose_buffer_msg = Skeleton3DBuffer()
                    array_msg =  Float64MultiArray()
                    output_buffer = get_3dpose_in_camera_frame(np.array(pose_buffer[-k*output_size:]), Transform=W_to_C_T)
                    print(output_buffer.shape)
                    output_buffer_shape = output_buffer.shape
                    array_msg.data = output_buffer.flatten()
                    # array_msg.layout.data_offset =  0 # no padding
                    # dim = []
                    # dim.append(MultiArrayDimension("points", n, 3*n))
                    # dim.append(MultiArrayDimension("coords", 3, 1))
                    # array_msg.layout.dim = dim

                    pose_buffer_msg.skeleton_3d_17_flat = array_msg
                    pose_buffer_msg.shape = output_buffer_shape 
                    publisher.publish(pose_buffer_msg)
            

            # Update GL view
            # viewer.update_view(image, bodies) 
            # # Update OCV view
            # image_left_ocv = image.get_data() 
            # cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            # cv2.imshow("ZED | 2D View", image_left_ocv)
            # cv2.waitKey(10)

    # viewer.exit()

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()

def get_point_clouds(body, W_to_C_T, br, relative_to="world"):
    
    if relative_to == "world": 
        skeleton_3d = body.keypoint
    elif relative_to == "camera":
        # local_position_per_joint
        skeleton_3d = body.keypoint
        skeleton_3d = np.hstack((skeleton_3d, np.ones((34,1))))
        skeleton_3d = np.matmul(W_to_C_T,np.transpose(skeleton_3d))
        skeleton_3d = np.transpose(skeleton_3d)


    # if not np.array_equal(skeleton_3d, body.keypoint - body.keypoint[0], equal_nan=False):
    #      skeleton_3d = body.keypoint - body.keypoint[0]

    pc = PointCloud()
    pc.header = Header()
    pc.header.frame_id = relative_to

    if len(skeleton_3d) > 0:
        # POSE_18
        # body_format == sl.BODY_FORMAT.POSE_18
        if len(skeleton_3d) == 18:
            pc.points[0] = Point32((skeleton_3d[8] + skeleton_3d[11])/2)
            pc.points[1].x,pc.points[1].y,pc.points[1].z = Keypoint3D(skeleton_3d[11])
            pc.points[2].x,pc.points[2].y,pc.points[2].z = Keypoint3D(skeleton_3d[12])
            pc.points[3].x = Keypoint3D(skeleton_3d[13])
            pc.points[4].x = Keypoint3D(skeleton_3d[8])
            pc.points[5].x = Keypoint3D(skeleton_3d[9])
            pc.points[6].x = Keypoint3D(skeleton_3d[10])
            pc.points[7].x = Keypoint3D((skeleton_3d[8]+skeleton_3d[11]+skeleton_3d[1])/3)
            pc.points[8].x = Keypoint3D(skeleton_3d[1])
            pc.points[9].x = Keypoint3D(skeleton_3d[0])
            pc.points[10].x = Keypoint3D((skeleton_3d[16] + skeleton_3d[17])/2)
            pc.points[11].x = Keypoint3D(skeleton_3d[2])
            pc.points[12].x = Keypoint3D(skeleton_3d[3]) 
            pc.points[13].x = Keypoint3D(skeleton_3d[4])
            pc.points[14].x = Keypoint3D(skeleton_3d[5])
            pc.points[15].x = Keypoint3D(skeleton_3d[6])
            pc.points[16].x = Keypoint3D(skeleton_3d[7])
            

        # POSE_34
        # body_format == sl.BODY_FORMAT.POSE_34
        elif len(skeleton_3d) == 34:
            pc.points.append(Point32(x=skeleton_3d[0][0],y=skeleton_3d[0][1],z=skeleton_3d[0][2]))
            pc.points.append(Point32(x=skeleton_3d[18][0],y=skeleton_3d[18][1],z=skeleton_3d[18][2]))
            pc.points.append(Point32(x=skeleton_3d[19][0],y=skeleton_3d[19][1],z=skeleton_3d[19][2]))
            pc.points.append(Point32(x=skeleton_3d[20][0],y=skeleton_3d[20][1],z=skeleton_3d[20][2]))
            pc.points.append(Point32(x=skeleton_3d[22][0],y=skeleton_3d[22][1],z=skeleton_3d[22][2]))
            pc.points.append(Point32(x=skeleton_3d[23][0],y=skeleton_3d[23][1],z=skeleton_3d[23][2]))
            pc.points.append(Point32(x=skeleton_3d[24][0],y=skeleton_3d[24][1],z=skeleton_3d[24][2]))
            pc.points.append(Point32(x=skeleton_3d[2][0],y=skeleton_3d[2][1],z=skeleton_3d[2][2]))
            pc.points.append(Point32(x=skeleton_3d[3][0],y=skeleton_3d[3][1],z=skeleton_3d[3][2]))
            pc.points.append(Point32(x=skeleton_3d[27][0],y=skeleton_3d[27][1],z=skeleton_3d[27][2]))
            head = (skeleton_3d[29]+skeleton_3d[31])/2
            pc.points.append(Point32(x=head[0],y=head[1],z=head[2]))
            pc.points.append(Point32(x=skeleton_3d[12][0],y=skeleton_3d[12][1],z=skeleton_3d[12][2]))
            pc.points.append(Point32(x=skeleton_3d[13][0],y=skeleton_3d[13][1],z=skeleton_3d[13][2]))
            pc.points.append(Point32(x=skeleton_3d[14][0],y=skeleton_3d[14][1],z=skeleton_3d[14][2]))
            pc.points.append(Point32(x=skeleton_3d[5][0],y=skeleton_3d[5][1],z=skeleton_3d[5][2]))
            pc.points.append(Point32(x=skeleton_3d[6][0],y=skeleton_3d[6][1],z=skeleton_3d[6][2]))
            pc.points.append(Point32(x=skeleton_3d[7][0],y=skeleton_3d[7][1],z=skeleton_3d[7][2]))
        
        # print("hip translation", skeleton_3d[0,:3])
        # br.sendTransform( skeleton_3d[0,:3], # translation
        #                     (0,0,0,1),# tr.quaternion_from_matrix(h2c_R), # bodies.object_list[0].global_root_orientation, # (x, y, z, w) rotation
        #                     rospy.Time.now(),
        #                     'skeleton_17_pc',
        #                     'camera')
    return pc


def zed32_to_17_format(skeleton_3d):
    skeleton_3d_17 = np.zeros((17,3))
    if len(skeleton_3d) > 0:
        # POSE_18s
        # body_format == sl.BODY_FORMAT.POSE_18
        if len(skeleton_3d) == 18:
            skeleton_3d_17[0,:] = (skeleton_3d[8] + skeleton_3d[11])/2
            skeleton_3d_17[1,:] = skeleton_3d[11]
            skeleton_3d_17[2,:] = skeleton_3d[12]
            skeleton_3d_17[3,:] = skeleton_3d[13]
            skeleton_3d_17[4,:] = skeleton_3d[8]
            skeleton_3d_17[5,:] = skeleton_3d[9]
            skeleton_3d_17[6,:] = skeleton_3d[10]
            skeleton_3d_17[7,:] = (skeleton_3d[8]+skeleton_3d[11]+skeleton_3d[1]/3)
            skeleton_3d_17[8,:] = skeleton_3d[1]
            skeleton_3d_17[9,:] = skeleton_3d[0]
            skeleton_3d_17[10,:] = (skeleton_3d[16] + skeleton_3d[17]/2)
            skeleton_3d_17[11,:] = skeleton_3d[2]
            skeleton_3d_17[12,:] = skeleton_3d[3] 
            skeleton_3d_17[13,:] = skeleton_3d[4]
            skeleton_3d_17[14,:] = skeleton_3d[5]
            skeleton_3d_17[15,:] = skeleton_3d[6]
            skeleton_3d_17[16,:] = skeleton_3d[7]
            

        # POSE_34
        # body_format == sl.BODY_FORMAT.POSE_34
        elif len(skeleton_3d) == 34:
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
            new_indices = [0,18,19,20,22,23,24,2,3,26,27,12,13,14,5,6,7]
            skeleton_3d_17 = skeleton_3d[new_indices]

    return skeleton_3d_17

    
def get_3dpose_in_camera_frame(skeleton_buffer, Transform):
    # skeleton_3d = body.keypoint
    # test
    # a = time.time()
    # s3 = []
    # for i in range(skeleton_buffer.shape[0]):
    #     skeleton_3d = skeleton_buffer[i,:,:]
    #     skeleton_3d = np.hstack((skeleton_3d, np.ones((skeleton_3d.shape[0],1))))
    #     skeleton_3d = np.matmul(Transform,np.transpose(skeleton_3d))
    #     skeleton_3d = np.transpose(skeleton_3d)
    #     s3.append(skeleton_3d)
    
    
    # s3 = np.array(s3)
    # print(time.time()-a)

    # b=time.time()
    skeleton_buffer = np.concatenate((skeleton_buffer, np.ones((skeleton_buffer.shape[0],skeleton_buffer.shape[1],1))),axis=2)
    skeleton_buffer_shape = skeleton_buffer.shape
    skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0]*skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
    skeleton_buffer_t = np.transpose(skeleton_buffer)
    skeleton_buffer_t = np.matmul(Transform, skeleton_buffer_t)
    skeleton_buffer = np.transpose(skeleton_buffer_t)
    skeleton_buffer = np.reshape(skeleton_buffer, (skeleton_buffer_shape[0],skeleton_buffer_shape[1],skeleton_buffer_shape[2]))
    # print(time.time()-b)
    
    return skeleton_buffer[:,:,:3]


if __name__ == "__main__":
    pose3d_zed()