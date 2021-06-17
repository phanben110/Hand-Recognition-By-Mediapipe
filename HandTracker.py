import numpy as np
from collections import namedtuple
import mediapipe_utils as mpu
import depthai as dai
import cv2
from pathlib import Path
from FPS import FPS, now
import argparse
import BEN_DrawFinger as DrawFinger

#define 
DrawFinger = DrawFinger.DrawFinger( True )     

# def to_planar(arr: np.ndarray, shape: tuple) -> list:
def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2,0,1)



class HandTracker:
    def __init__(self, input_file=None,
                pd_path="models/palm_detection.blob", 
                pd_score_thresh=0.7, pd_nms_thresh=0.3,
                use_lm=True,
                lm_path="models/hand_landmark.blob",
                lm_score_threshold=0.7,
                recPath="models/torchHandRec.blob",
                recScore = 0.8,
                useRec=True):

        self.camera = input_file is None
        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        #self.use_gesture = use_gesture 
        self.useRec  = useRec 
        self.recPath = recPath
        self.recScore = recScore

        #that is labels 

        self.labels =  ['Ok', 'Silent', 'Dislike', 'Like', 'Hi', 'Hello', 'Stop' , ' ' ]


        # this custom function 
        self.pointFinger = [] 
        
        if not self.camera:
            if input_file.endswith('.jpg') or input_file.endswith('.png') :
                self.image_mode = True
                self.img = cv2.imread(input_file)
                self.video_size = np.min(self.img.shape[:2])
            else:
                self.image_mode = False
                self.cap = cv2.VideoCapture(input_file)
                width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
                height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
                self.video_size = int(min(width, height))
        
        # Create SSD anchors 
        # https://github.com/google/mediapipe/blob/master/mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt
        anchor_options = mpu.SSDAnchorOptions(num_layers=4, 
                                min_scale=0.1484375,
                                max_scale=0.75,
                                input_size_height=128,
                                input_size_width=128,
                                anchor_offset_x=0.5,
                                anchor_offset_y=0.5,
                                strides=[8, 16, 16, 16],
                                aspect_ratios= [1.0],
                                reduce_boxes_in_lowest_layer=False,
                                interpolated_scale_aspect_ratio=1.0,
                                fixed_anchor_size=True)
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]
        print(f"{self.nb_anchors} anchors have been created")

        # Rendering flags
        if self.use_lm:
            self.show_pd_box = False
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_handedness = False
            self.show_landmarks = False
            self.show_scores = False
            self.useRec = True
        else:
            self.show_pd_box = True
            self.show_pd_kps = False
            self.show_rot_rect = False
            self.show_scores = False
        

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(version = dai.OpenVINO.Version.VERSION_2021_2)
        self.pd_input_length = 128

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            cam = pipeline.createColorCamera()
            cam.setPreviewSize(self.pd_input_length, self.pd_input_length)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            # Crop video to square shape (palm detection takes square image as input)
            self.video_size = min(cam.getVideoSize())
            cam.setVideoSize(self.video_size, self.video_size)
            cam.setFps(50)
            cam.setInterleaved(False)
            cam.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_out = pipeline.createXLinkOut()
            cam_out.setStreamName("cam_out")
            # Link video output to host for higher resolution
            cam.video.link(cam_out.input)

        # Define palm detection model
        print("Creating Palm Detection Neural Network...")
        pd_nn = pipeline.createNeuralNetwork()
        pd_nn.setBlobPath(str(Path(self.pd_path).resolve().absolute()))
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        # Specify that network takes latest arriving frame in non-blocking manner
        # Palm detection input                 
        if self.camera:
            pd_nn.input.setQueueSize(1)
            pd_nn.input.setBlocking(False)
            cam.preview.link(pd_nn.input)
        else:
            pd_in = pipeline.createXLinkIn()
            pd_in.setStreamName("pd_in")
            pd_in.out.link(pd_nn.input)
        # Palm detection output
        pd_out = pipeline.createXLinkOut()
        pd_out.setStreamName("pd_out")
        pd_nn.out.link(pd_out.input)
        

         # Define hand landmark model
        if self.use_lm:
            print("Creating Hand Landmark Neural Network...")          
            lm_nn = pipeline.createNeuralNetwork()
            lm_nn.setBlobPath(str(Path(self.lm_path).resolve().absolute()))
            lm_nn.setNumInferenceThreads(1)
            # Hand landmark input
            self.lm_input_length = 224
            lm_in = pipeline.createXLinkIn()
            lm_in.setStreamName("lm_in")
            lm_in.out.link(lm_nn.input)
            # Hand landmark output
            lm_out = pipeline.createXLinkOut()
            lm_out.setStreamName("lm_out")
            lm_nn.out.link(lm_out.input)

        if self.useRec: 
            detection_nn = pipeline.createNeuralNetwork()
            detection_nn.setBlobPath(str(Path(self.recPath).resolve().absolute()))
            #create viture image input for neural network
            imgNN = pipeline.createXLinkIn()
            imgNN.setStreamName("in_nn")
            imgNN.out.link(detection_nn.input)
            
            # Create outputs
            xout_nn = pipeline.createXLinkOut()
            xout_nn.setStreamName("nn")
            detection_nn.out.link(xout_nn.input)



        print("Pipeline created.")
        return pipeline        

    
    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16) # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape((self.nb_anchors,18)) # 896x18
        # Decode bboxes
        self.regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        self.regions = mpu.non_max_suppression(self.regions, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(self.regions)
            mpu.rect_transformation(self.regions, self.video_size, self.video_size)

    def pd_render(self, frame):
        for r in self.regions:
            if self.show_pd_box:
                box = (np.array(r.pd_box) * self.video_size).astype(int)
                cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0,255,0), 2)
            if self.show_pd_kps:
                for i,kp in enumerate(r.pd_kps):
                    x = int(kp[0] * self.video_size)
                    y = int(kp[1] * self.video_size)
                    cv2.circle(frame, (x, y), 6, (0,0,255), -1)
                    cv2.putText(frame, str(i), (x, y+12), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,0), 2)
            if self.show_scores:
                cv2.putText(frame, f"Palm score: {r.pd_score:.2f}", 
                        (int(r.pd_box[0] * self.video_size+10), int((r.pd_box[1]+r.pd_box[3])*self.video_size+60)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

            
    def lm_postprocess(self, region, inference):
        region.lm_score = inference.getLayerFp16("Identity_1")[0]    
        region.handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add"))
        
        lm = []
        for i in range(int(len(lm_raw)/3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3*i:3*(i+1)]/self.lm_input_length)
        region.landmarks = lm


    
    def lm_render(self, frame, region):
        if region.lm_score > self.lm_score_threshold:
            if self.show_rot_rect:
                cv2.polylines(frame, [np.array(region.rect_points)], True, (0,255,255), 2, cv2.LINE_AA)
                
            src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
            dst = np.array([ (x, y) for x,y in region.rect_points[1:]], dtype=np.float32) # region.rect_points[0] is left bottom point !
            mat = cv2.getAffineTransform(src, dst) 
            lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in region.landmarks]), axis=0)
            lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)
                
            check,box, img = DrawFinger.drawAndResize(frame,lm_xy,size = 100  )

            if check: 
                
                cv2.imshow("crop", img  )
                img = cv2.resize(img ,(26,26))
                imgCrop = img.T  
                cv2.rectangle( frame , ( box[0] - box[4] , box[1] - box[4] ) , ( box[2] + box[4] , box[3]+ box[4]  ) , (0,255,0),2 )
            else: 
                imgCrop = None 

                #check, cropImg = DrawFinger.drawAndResize(frame,lm_xy,size = 100 , draw = True ) 
                #cv2.polylines(frame, lines, False, (255,255, 255), 12, cv2.LINE_AA)
            if self.show_landmarks: 
                    
                list_connections = [[0, 1, 2, 3, 4], 
                                    [0, 5, 6, 7, 8], 
                                    [5, 9, 10, 11, 12],
                                    [9, 13, 14 , 15, 16],
                                    [13, 17],
                                    [0, 17, 18, 19, 20]]
                # TODO lm_xy is a variable to store the point store 
                print (len ( lm_xy )) 
                lines = [np.array([lm_xy[point] for point in line]) for line in list_connections]
                #print ( f"print line {lines}" ) 
                #print ( len(lines)) 
                # this function to crop image into small frame  
                cv2.polylines(frame, lines, False, (255,255, 255), 12, cv2.LINE_AA)
                for x,y in lm_xy:
                    cv2.circle(frame, (x, y), 6, (255,255,255), -1)

            if self.show_handedness:
                cv2.putText(frame, f"RIGHT {region.handedness:.2f}" if region.handedness > 0.5 else f"LEFT {1-region.handedness:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+20)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0) if region.handedness > 0.5 else (0,0,255), 2)
            if self.show_scores:
                cv2.putText(frame, f"Landmark score: {region.lm_score:.2f}", 
                        (int(region.pd_box[0] * self.video_size+10), int((region.pd_box[1]+region.pd_box[3])*self.video_size+90)), 
                        cv2.FONT_HERSHEY_PLAIN, 2, (255,255,0), 2)

            return imgCrop

    
    def softmax(self, x): 
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)


    def recProcess(self,imgCrop ,  q_rec_in , q_rec_out) :
        #print ( imgCrop.size )
        try:
            nn_data = dai.NNData()
            nn_data.setLayer("0", imgCrop )
            q_rec_in.send(nn_data)
            in_nn = q_rec_out.tryGet()

            if in_nn is not None:
                data = self.softmax(in_nn.getFirstLayerFp16())
                result_conf = np.max(data)
                return result_conf , data 
            else : 
                return 0 , 0 
        except: 
            return 0,0

    def recRender(self, frame , result_conf , data ):
        if result_conf > 0.95: 
            label = self.labels[int ( np.argmax(data))]  
            conf = f"{round(100*result_conf,2)}%"
            print ( conf ) 
            cv2.putText( frame , f"Status: {label} {conf}" , (200,26) , cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255  ) ,thickness=3)
            
        else: 
            label = ' '
            conf = ' '
            cv2.putText( frame , f"Status: {label} {conf}" , (200,26) , cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 255  ) ,thickness=3)
 

        


    def run(self):

        device = dai.Device(self.create_pipeline())
        device.startPipeline()

        # Define data queues 
        if self.camera:
            q_video = device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=1, blocking=False)

            if self.use_lm:
                q_lm_out = device.getOutputQueue(name="lm_out", maxSize=2, blocking=False)
                q_lm_in = device.getInputQueue(name="lm_in")

        else:
            q_pd_in = device.getInputQueue(name="pd_in")
            q_pd_out = device.getOutputQueue(name="pd_out", maxSize=4, blocking=True)
            if self.use_lm:
                q_lm_out = device.getOutputQueue(name="lm_out", maxSize=4, blocking=True)
                q_lm_in = device.getInputQueue(name="lm_in")

        if self.useRec: 
            q_rec_in = device.getInputQueue("in_nn")
            q_rec_out = device.getOutputQueue(name="nn", maxSize=1, blocking=False)

        self.fps = FPS(mean_nb_frames=20)

        seq_num = 0
        nb_pd_inferences = 0
        nb_lm_inferences = 0
        glob_pd_rtrip_time = 0
        glob_lm_rtrip_time = 0
        while True:
            self.fps.update()
            if self.camera:
                in_video = q_video.get()
                video_frame = in_video.getCvFrame()
            else:
                if self.image_mode:
                    vid_frame = self.img
                else:
                    ok, vid_frame = self.cap.read()
                    if not ok:
                        break
                h, w = vid_frame.shape[:2]
                dx = (w - self.video_size) // 2
                dy = (h - self.video_size) // 2
                video_frame = vid_frame[dy:dy+self.video_size, dx:dx+self.video_size]
                frame_nn = dai.ImgFrame()
                frame_nn.setSequenceNum(seq_num)
                frame_nn.setWidth(self.pd_input_length)
                frame_nn.setHeight(self.pd_input_length)
                frame_nn.setData(to_planar(video_frame, (self.pd_input_length, self.pd_input_length)))
                q_pd_in.send(frame_nn)
                pd_rtrip_time = now()

                seq_num += 1

            annotated_frame = video_frame.copy()

            # Get palm detection
            inference = q_pd_out.get()
            if not self.camera: glob_pd_rtrip_time += now() - pd_rtrip_time
            self.pd_postprocess(inference)
            self.pd_render(annotated_frame)
            nb_pd_inferences += 1

            # Hand landmarks
            if self.use_lm:
                for i,r in enumerate(self.regions):
                    img_hand = mpu.warp_rect_img(r.rect_points, video_frame, self.lm_input_length, self.lm_input_length)
                    nn_data = dai.NNData()   
                    nn_data.setLayer("input_1", to_planar(img_hand, (self.lm_input_length, self.lm_input_length)))
                    q_lm_in.send(nn_data)
                    if i == 0: lm_rtrip_time = now() # We measure only for the first region
                
                # Retrieve hand landmarks
                for i,r in enumerate(self.regions):
                    inference = q_lm_out.get()
                    if i == 0: glob_lm_rtrip_time += now() - lm_rtrip_time
                    self.lm_postprocess(r, inference)
                    imgCrop = self.lm_render(annotated_frame, r)
                    nb_lm_inferences += 1

            # Hand recognitions 
                    if self.useRec: 
                        
                        result, data  = self.recProcess(imgCrop , q_rec_in, q_rec_out) 
                        self.recRender( annotated_frame,result, data )  

                
            self.fps.display(annotated_frame, orig=(50,50),color=(240,180,100))
            cv2.imshow("video", annotated_frame)

            key = cv2.waitKey(1) 
            if key == ord('q') or key == 27:
                break
            elif key == 32:
                # Pause on space bar
                cv2.waitKey(0)
            elif key == ord('1'):
                self.show_pd_box = not self.show_pd_box
            elif key == ord('2'):
                self.show_pd_kps = not self.show_pd_kps
            elif key == ord('3'):
                self.show_rot_rect = not self.show_rot_rect
            elif key == ord('4'):
                self.show_landmarks = not self.show_landmarks
            elif key == ord('5'):
                self.show_handedness = not self.show_handedness
            elif key == ord('6'):
                self.show_scores = not self.show_scores

        # Print some stats
        if not self.camera:
            print(f"# video files frames                 : {seq_num}")
            print(f"# palm detection inferences received : {nb_pd_inferences}")
            print(f"# hand landmark inferences received  : {nb_lm_inferences}")
            print(f"Palm detection round trip            : {glob_pd_rtrip_time/nb_pd_inferences*1000:.1f} ms")
            print(f"Hand landmark round trip             : {glob_lm_rtrip_time/nb_lm_inferences*1000:.1f} ms")
           

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, 
                        help="Path to video or image file to use as input (if not specified, use OAK color camera)")
    parser.add_argument("--pd_m", default="models/palm_detection.blob", type=str,
                        help="Path to a blob file for palm detection model (default=%(default)s)")
    parser.add_argument('--no_lm', action="store_true", 
                        help="only the palm detection model is run, not the hand landmark model")
    parser.add_argument("--lm_m", default="models/hand_landmark.blob", type=str,
                        help="Path to a blob file for landmark model (default=%(default)s)")
    parser.add_argument('-rec', '--rec', action="store_true", 
                        help="enable gesture recognition")
    args = parser.parse_args()

    ht = HandTracker(input_file=args.input, pd_path=args.pd_m, use_lm= not args.no_lm, lm_path=args.lm_m, useRec=args.rec)
    ht.run()
