# coding=utf-8
from time import time
import cv2
import numpy as np
import os
from modules.DepthAI import DepthAI
from utils import timer, run_nn, to_planar, to_nn_result
from utils import mediapipe as mpu
from utils.BEN_DrawFinger import DrawFinger


class HandGesture(DepthAI):
    def __init__(self, camera=False,
                no_debug=True,
                pd_path="models/palm_detection.blob",
                pd_score_thresh=0.7, pd_nms_thresh=0.3,  # defaut = 0.3
                use_lm=True,
                lm_path="models/hand_landmark.blob",
                lm_score_threshold=0.6,
                lm_input_length=224,
                rec_path="models/model12ClassSize26.blob",
                size_rec=26,
                rec_score=0.9,
                frame_pass=0,
                use_rec=True):
        self.cam_size = (900, 900)
        # self.cv_status_code = os.environ["CV_STATUS"]

        self.pd_path = pd_path
        self.pd_score_thresh = pd_score_thresh
        self.pd_nms_thresh = pd_nms_thresh
        self.use_lm = use_lm
        self.lm_path = lm_path
        self.lm_score_threshold = lm_score_threshold
        self.use_rec = use_rec
        self.rec_path = rec_path
        self.rec_score = rec_score
        self.size_rec = size_rec
        self.frame_pass = frame_pass
        self.lm_input_length = lm_input_length
        super(HandGesture, self).__init__(
            camera, not no_debug)

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
        self.labels = ['Three','Stop','Hello','Zero','Tym','Nothing','Two','One','Dislike', 'Like' ,'Four','Ok' ,'Nothing' ]
        self.anchors = mpu.generate_anchors(anchor_options)
        self.nb_anchors = self.anchors.shape[0]


    def create_nns(self):
        if self.camera:
            self.create_nn(
                model_path=self.pd_path,
                model_name="palm",
                first=self.camera,
                input_shape=(128, 128)
            )

        else:
            self.create_nn(
                model_path=self.pd_path,
                model_name="palm"
            )

        self.create_nn(
            model_path=self.lm_path,
            model_name="landmark"
        )

        self.create_nn(
            model_path=self.rec_path,
            model_name="classify"
        )

    def start_nns(self):
        if self.camera:
            self.palm_nn = self.device.getOutputQueue("palm_nn", 4, False)
        else:
            self.palm_in = self.device.getInputQueue("palm_in")
            self.palm_nn = self.device.getOutputQueue("palm_nn", 4, False)

        self.landmark_in = self.device.getInputQueue("landmark_in")
        self.landmark_nn = self.device.getOutputQueue("landmark_nn")

        self.classify_in = self.device.getInputQueue("classify_in")
        self.classify_nn = self.device.getOutputQueue("classify_nn")

    @timer
    def run_palm_detection(self, image_frame= None):
        if self.camera:
            image_frame = self.frame
            nn_data = self.palm_nn.tryGet()
        else:
            if image_frame is None:
                raise ValueError("image_frame is None")
            send_data = {
                "data": to_planar(image_frame, (128, 128))
            }
            nn_data = run_nn(
                self.palm_in, self.palm_nn, send_data
            )
        if nn_data is None:
            return (False, None)
        else:
            regions = self.pd_postprocess(inference=nn_data)
            return (True, regions)

    def pd_postprocess(self, inference):
        scores = np.array(inference.getLayerFp16("classificators"), dtype=np.float16)  # 896
        bboxes = np.array(inference.getLayerFp16("regressors"), dtype=np.float16).reshape(
            (self.nb_anchors, 18))  # 896x18
        # Decode bboxes
        regions = mpu.decode_bboxes(self.pd_score_thresh, scores, bboxes, self.anchors)
        # Non maximum suppression
        regions = mpu.non_max_suppression(regions, self.pd_nms_thresh)
        if self.use_lm:
            mpu.detections_to_rect(regions)
            mpu.rect_transformation(regions, self.cam_size[0], self.cam_size[1])
        return regions

    @timer
    def run_landmark(self, image_frame, region):
        img_hand = mpu.warp_rect_img(region.rect_points, image_frame, self.lm_input_length, self.lm_input_length)
        send_data = {
            "input_1": to_planar(img_hand, (self.lm_input_length, self.lm_input_length))
        }
        nn_data = run_nn(self.landmark_in, self.landmark_nn, send_data)
        if nn_data is None:
            return (False, None)
        lm_score, handedness, lm = self.lm_postprocess(nn_data)
        if lm_score < self.lm_score_threshold:
            return (False, lm)
        return (True, lm)

    def lm_postprocess(self, inference):
        lm_score = inference.getLayerFp16("Identity_1")[0]
        handedness = inference.getLayerFp16("Identity_2")[0]
        lm_raw = np.array(inference.getLayerFp16("Identity_dense/BiasAdd/Add"))
        lm = []
        for i in range(int(len(lm_raw) / 3)):
            # x,y,z -> x/w,y/h,z/w (here h=w)
            lm.append(lm_raw[3 * i:3 * (i + 1)] / self.lm_input_length)
        return lm_score, handedness, lm

    def lm_render(self, frame, region, landmarks):
        if self.debug:
            cv2.polylines(self.debug_frame, [np.array(region.rect_points)], True, (0, 255, 255), 2, cv2.LINE_AA)

        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array([(x, y) for x, y in region.rect_points[1:]],
                       dtype=np.float32)  # region.rect_points[0] is left bottom point !
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(np.array([(l[0], l[1]) for l in landmarks]), axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat)).astype(np.int)

        check, box, img = DrawFinger().drawAndResize(img= frame,point= lm_xy, size=100)
        if check:
            img = cv2.resize(img, (self.size_rec, self.size_rec))
            if self.debug:
                cv2.imshow("lm_crop", img)
            img_crop = img.T
            img_crop.reshape(1, 1, self.size_rec, self.size_rec)

        else:
            img_crop = None
            box = None

        return img_crop, box

    @timer
    def run_classify(self, image_crop):
        send_data = {
            "0": image_crop
        }
        nn_data = run_nn(self.classify_in, self.classify_nn, send_data)
        if nn_data is None:
            return (False, None, None)
        else:
            data = self.softmax(nn_data.getFirstLayerFp16())
            result_conf = np.max(data)
            label = self.labels[int(np.argmax(data))]
            return (True, result_conf, label)

    def run_camera(self):
        # start FPS monitor
        self.start()
        while self.camera:
            # get frame from camera
            in_rgb = self.preview.tryGet()
            if in_rgb is not None:

                shape = (3, in_rgb.getHeight(), in_rgb.getWidth())
                self.frame = (
                    in_rgb.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
                )
                self.frame = np.ascontiguousarray(self.frame)
                try:
                    self.parse()
                except StopIteration:
                    break

    def run(self):
        self.warm_up()
        hand_signal = self.run_camera()

    def parse(self):
        # if self.debug:
        self.debug_frame = self.frame.copy()

        self.parse_action()

        # update camera FPS
        self.fps_cam.update()

        # if self.debug:
        self.display()

    def parse_action(self, image_frame=None):
        # run palm detection
        if image_frame is None:
            image_frame = self.frame
            (success, regions), pd_runtime = self.run_palm_detection()
        else:
            (success, regions), pd_runtime = self.run_palm_detection(image_frame)

        # run landmarks detection
        if success:
            print(regions)
            for region in regions:
                (success, lm), lm_runtime = self.run_landmark(image_frame, region)

                # run classification
                print(lm)
                if success:
                    img_crop, box = self.lm_render(image_frame, region, lm)
                    if img_crop is None:
                        print("img_crop is None")
                    else:
                        (success, result_conf, label), classify_runtime = self.run_classify(img_crop)
                        if success:
                            if result_conf > self.rec_score:
                                self.put_text(
                                    f"conf:{result_conf:.2f} - {label}%",
                                    (box[0] - box[4], box[1] - box[4] - 10),
                                    (244, 0, 255),
                                )

    def display(self):
        cv2.imshow("debug_frame", self.debug_frame)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            self.fps_cam.stop()
            self.fps_nn.stop()
            print(
                f"FPS_CAMERA: {self.fps_cam.fps():.2f} , FPS_NN: {self.fps_nn.fps():.2f}"
            )
            raise StopIteration()

    def softmax(self, x): 
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
