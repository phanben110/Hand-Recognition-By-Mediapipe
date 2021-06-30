import cv2
import math
import numpy as np
import yaml

def face_orientation(frame, image_points):
    size = frame.shape #(height, width, color_channel)    

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    # imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    # modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return (-float(roll), -float(yaw))

def get_headpose(lm_points, frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    image_points = np.array([
        (lm_points[30][0] - x_min, lm_points[30][1] - y_min),   # Nose tip
        (lm_points[8][0] - x_min, lm_points[8][1] - y_min),     # Chin
        (lm_points[36][0] - x_min, lm_points[36][1] - y_min),   # Left eye left corner
        (lm_points[45][0] - x_min, lm_points[45][1] - y_min),   # Right eye right corner
        (lm_points[48][0] - x_min, lm_points[48][1] - y_min),   # Left Mouth corner
        (lm_points[64][0] - x_min, lm_points[64][1] - y_min),   # Right mouth corner
    ], dtype="double")


    roll, yaw = face_orientation(frame, image_points)
    # calc roll
    x1, y1 = image_points[2]
    x2, y2 = image_points[3]
    huyen = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    doi = y2 - y1
    sin = doi / huyen
    roll = np.arcsin(sin) / np.pi * 180
    #
    return (roll, yaw)
