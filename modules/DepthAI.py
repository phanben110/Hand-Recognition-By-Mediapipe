# coding=utf-8
import os
import shutil
from pathlib import Path
import threading
import time
import cv2
import depthai
import numpy as np
from imutils.video import FPS
import unidecode


class DepthAI:
    def __init__(
            self,
            camera=False,
            images=None,
            debug=False,
            is_db=False,
    ):
        print("Loading pipeline...")
        self.camera = camera
        self.images = images
        self.fps_cam = FPS()
        self.fps_nn = FPS()
        self.debug = debug
        self.is_db = is_db
        self.create_pipeline()
        # self.warm_up()
        self.fontScale = 1 if self.camera else 2
        self.lineType = 0 if self.camera else 3

    def create_pipeline(self):
        print("Creating pipeline...")
        self.pipeline = depthai.Pipeline()

        if self.camera:
            # ColorCamera
            print("Creating Color Camera...")
            self.cam = self.pipeline.createColorCamera()
            self.cam.setPreviewSize(self._cam_size[1], self._cam_size[0])
            self.cam.setResolution(
                depthai.ColorCameraProperties.SensorResolution.THE_4_K
            )
            self.cam.setInterleaved(False)
            self.cam.setBoardSocket(depthai.CameraBoardSocket.RGB)
            self.cam.setColorOrder(depthai.ColorCameraProperties.ColorOrder.BGR)

            self.cam_xout = self.pipeline.createXLinkOut()
            self.cam_xout.setStreamName("preview")
            self.cam.preview.link(self.cam_xout.input)

        self.create_nns()

        print("Pipeline created.")

    def create_nns(self):
        pass

    def create_nn(self, model_path: str, model_name: str, first: bool = False, input_shape: tuple = None):
        """

        :param input_shape: model input shape
        :param model_path: model path
        :param model_name: model abbreviation
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createNeuralNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.input.setBlocking(False)
        if first and self.camera:
            if input_shape is None:
                raise ValueError("input_shape can not be None when first is True")
            if self._cam_size[1] != input_shape[1] or self._cam_size[0] != input_shape[0]:
                # Create face detection resize manip
                detection_manip = self.pipeline.createImageManip()
                detection_manip.setResize(input_shape[0], input_shape[1])

                # Link camera output to detection_manip input
                self.cam.preview.link(detection_manip.inputImage)

                # Link to face detection neural network input
                detection_manip.out.link(model_nn.input)
            else:
                print("linked cam.preview to model_nn.input")
                self.cam.preview.link(model_nn.input)

        else:
            model_in = self.pipeline.createXLinkIn()
            model_in.setStreamName(f"{model_name}_in")
            model_in.out.link(model_nn.input)

        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def create_mobilenet_nn(
            self,
            model_path: str,
            model_name: str,
            conf: float = 0.5,
            first: bool = False,
            input_shape: tuple = None,
    ):
        """
        :param input_shape: model input shape
        :param model_path: model name
        :param model_name: model abbreviation
        :param conf: confidence threshold
        :param first: Is it the first model
        :return:
        """
        # NeuralNetwork
        print(f"Creating {model_path} Neural Network...")
        model_nn = self.pipeline.createMobileNetDetectionNetwork()
        model_nn.setBlobPath(str(Path(f"{model_path}").resolve().absolute()))
        model_nn.setConfidenceThreshold(conf)
        model_nn.input.setBlocking(False)
        model_in = self.pipeline.createXLinkIn()
        model_in.setStreamName(f"{model_name}_in")
        if first:
            if input_shape is None:
                raise ValueError("input_shape can not be None when first is True")

            if self._cam_size[1] != input_shape[1] or self._cam_size[0] != input_shape[0]:
                # Create face detection resize manip
                detection_manip = self.pipeline.createImageManip()
                detection_manip.setResize(input_shape[0], input_shape[1])

                # Link camera output to detection_manip input
                self.cam.preview.link(detection_manip.inputImage)

                # Link to face detection neural network input
                detection_manip.out.link(model_nn.input)
            else:
                # Link camera output to detection neural nerwork
                self.cam.preview.link(model_nn.input)
        else:
            model_in.out.link(model_nn.input)
        model_nn_xout = self.pipeline.createXLinkOut()
        model_nn_xout.setStreamName(f"{model_name}_nn")
        model_nn.out.link(model_nn_xout.input)

    def start_pipeline(self):
        start_time = time.time()
        self.device = depthai.Device(self.pipeline)
        print("Starting pipeline...")
        if not self.device.startPipeline():
            raise ValueError("Start Pipeline fail")
        else:
            print("Start Pipeline successfull")
            device_memory_usage = self.device.getDdrMemoryUsage()
            print("Device memory used: {} - Remaining: {} - Total: {}\n".format(
                device_memory_usage.used,
                device_memory_usage.remaining,
                device_memory_usage.total))

        self.start_nns()

        if self.camera:
            self.preview = self.device.getOutputQueue(
                name="preview", maxSize=4, blocking=False
            )
        print("Start pipeline duration: ", time.time() - start_time)

    def stop_pipeline(self):
        self.device.close()
        del self.device

    def start_nns(self):
        pass

    def warm_up(self):
        self.start_pipeline()
        pass

    def put_text(self, text, dot, color=(0, 0, 255), font_scale=None, line_type=None):
        text = unidecode.unidecode(text)
        font_scale = font_scale if font_scale else self.fontScale
        line_type = line_type if line_type else self.lineType
        dot = tuple(dot[:2])
        cv2.putText(
            img=self.debug_frame,
            text=text,
            org=dot,
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=font_scale,
            color=color,
            lineType=line_type,
        )

    def draw_bbox(self, bbox, color):
        cv2.rectangle(
            img=self.debug_frame,
            pt1=(bbox[0], bbox[1]),
            pt2=(bbox[2], bbox[3]),
            color=color,
            thickness=2,
        )

    def draw_landmarks(self, landmasks, color=(0, 0, 255)):
        for counter, point in enumerate(landmasks):
            cv2.circle(img=self.debug_frame, center=point, radius=1, color=color)
            self.put_text(str(counter), point, font_scale=0.3)

    def draw_runtime(self, data: dict):
        dot = [10, 30]
        for key in data.keys():
            line = key.split("_runtime")[0] + ": " + str(round(data[key], 5))
            self.put_text(line, dot)
            dot = [dot[0], dot[1] + 30 * self.fontScale]

    @property
    def cam_size(self):
        return self._cam_size

    @cam_size.setter
    def cam_size(self, v):
        self._cam_size = v

    def start(self):
        self.fps_cam.start()
        self.fps_nn.start()

    def stop(self):
        del self.device

    def reset_action(self):
        pass

    def switch2cam(self):
        # TODO implement
        pass

    def switch2img(self):
        # TODO implement
        pass
