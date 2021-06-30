import time
import cv2
import datetime

import numpy
import numpy as np
import os
import depthai
from pathlib import Path

from functools import wraps

TIMEOUT = 2


def log_ouput(name, accuracy, record_time):
    dt = datetime.datetime.fromtimestamp(record_time).isoformat()
    log_msg = str(dt) + ": Name =  " + name + " - Acc = " + str(accuracy)
    print(log_msg)
    with open("logs.txt", "a", encoding="utf-8") as file:
        file.write(log_msg + "\n")


def log_test(real_name, name, accuracy):
    with open("logs_test.txt", "a", encoding="utf-8") as file:
        file.write("Real name = " + real_name + " - Output name = " + name + " - Acc = " + str(accuracy) + "\n")


def preprocess_gm_classify(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("~/Downloads/AloALo.png", img)
    img = np.array(img, dtype=np.float)
    img = img / 255
    return img


def preprocess_pcn_img(img, dim=None):
    if dim:
        img = cv2.resize(img, (dim, dim), interpolation=cv2.INTER_NEAREST)
    return img - np.array([104.0, 117.0, 123.0])


def to_planar(arr: np.ndarray, shape: tuple):
    return cv2.resize(arr, shape).transpose((2, 0, 1)).flatten()


def to_nn_result(nn_data):
    return np.array(nn_data.getFirstLayerFp16())


def run_nn(x_in, x_out, in_dict):
    nn_data = depthai.NNData()
    for key in in_dict:
        nn_data.setLayer(key, in_dict[key])
    x_in.send(nn_data)
    out = x_out.tryGet()
    start_time = time.time()
    while out is None:
        # print("Waiting For Message", end='\r')
        out = x_out.tryGet()
        if time.time() - start_time > TIMEOUT:
            break
    return out


def frame_norm(frame, *xy_vals):
    return (
            np.clip(np.array(xy_vals), 0, 1) * np.array(frame * (len(xy_vals) // 2))[::-1]
    ).astype(int)


def correction(frame, center=None, angle=None, invert=False):
    angle = int(angle)
    h, w = frame.shape[:2]
    if center is None:
        center = (h // 2, w // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1)
    affine = cv2.invertAffineTransform(mat).astype("float32")
    corr = cv2.warpAffine(
        frame,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
    )
    if invert:
        return corr, affine
    return corr


def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    similarity = np.dot(a, b.T) / (a_norm * b_norm)

    return similarity


def read_images(image_folder):
    data = dict()

    if not os.path.isdir(image_folder):
        raise Warning("Image folder not exist.")

    for r, d, f in os.walk(image_folder):
        for file in f:
            if '.jpg' in file or '.jpeg' in file:
                id = os.path.split(r)[-1]
                image = cv2.imread(str(Path(os.path.join(r, file)).resolve().absolute()))
                if id not in data.keys():
                    data[id] = []
                data[id].append(image)
    return data


def timer(func):
    """
    Return the runtime of the decorated function
    """

    @wraps(func)
    def wrapper_timmer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)

        # do someting after
        end_time = time.perf_counter()
        run_time = end_time - start_time
        # print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value, run_time

    return wrapper_timmer
