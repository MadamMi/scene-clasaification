import math
import random
import torch
import cv2
import numpy as np
from util.config import train_cfg as cfg


def _distort(image):

    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):

        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    else:
        # brightness distortion
        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # saturation distortion
        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        # hue distortion
        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        # contrast distortion
        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

    return image


def _mirror(image):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
    return image


def _resize_subtract_mean(image, insize, rgb_mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize), interpolation=interp_method)
    image = image.astype(np.float32)
    image -= rgb_mean
    return image.transpose(2, 0, 1)


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[:, 2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:, :2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0])*(box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0])*(box_b[:, 3]-box_b[:, 1]))
    union = area_a + area_b - inter
    return inter / union


def rotate(image):
    ro = random.random()
    ro_prob = cfg['rotate_prob']
    angle = cfg['rotate_angle']
    scale = 1

    if ro_prob < ro_prob:
        w = image.shape[1]
        h = image.shape[0]
        angle = np.random.uniform(0, angle)

        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height

        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        image = cv2.warpAffine(image, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    return image


def perspective(image):
    anglex = cfg['perspective_anglex']
    angley = cfg['perspective_angley']
    anglez = cfg['perspective_anglez']
    fov = cfg['perspective_fov']
    ppt_pro = cfg['perspective_prob']

    if ppt_pro < ppt_pro:

        img = image
        anglex = np.random.uniform(-anglex, anglex)
        angley = np.random.uniform(-angley, angley)
        anglez = np.random.uniform(-anglez, anglez)
        fov = np.random.uniform(-fov, fov)

        image, warpR = com(img, anglex, angley, anglez, fov)

    return image


def motion_blue(image):
    degree = cfg['motionblue_degree']
    angle = cfg['motionblue_angle']
    prob = cfg['motionblue_prob']

    image = np.array(image)
    degree = np.random.randint(1, degree)
    angle = np.random.uniform(0, angle)
    prob = random.random()
    # degree bigger, more blue
    if prob < prob:
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))

        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)

        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        image = np.array(blurred, dtype=np.uint8)

    return image


def gaussBlue(image):
    ksize = cfg['gaussBlue_kz']
    sigmax = cfg['gaussBlue_sigmax']
    sigmay = cfg['gaussBlue_sigmay']
    gauss_pro = cfg['gaussblue_prob']

    gauss_pro = random.random()
    if gauss_pro < gauss_pro:
        sigmax = np.random.randint(0, sigmax)
        sigmay = np.random.randint(0, sigmay)

        image = cv2.GaussianBlur(image, (ksize, ksize), sigmax, sigmay)

    return image


def com(img, anglex, angley, anglez, fov):
    h, w = img.shape[0:2]

    z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))

    rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                   [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                   [0, 1, 0, 0],
                   [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                   [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)), 0, 0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], np.float32)

    r = rx.dot(ry).dot(rz)

    pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

    p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
    p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
    p3 = np.array([0, h, 0, 0], np.float32) - pcenter
    p4 = np.array([w, h, 0, 0], np.float32) - pcenter

    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)

    list_dst = [dst1, dst2, dst3, dst4]
    #print('list dst : ', list_dst)
    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst = np.zeros((4, 2), np.float32)

    for i in range(4):
        dst[i, 0] = max((list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]), 0)
        dst[i, 1] = max((list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]), 0)

    # print("dst : ", dst)

    warpR = cv2.getPerspectiveTransform(org, dst)

    nw = max(max(max(dst[0][0], dst[1][0]), dst[2][0]), dst[3][0]) - \
         min(min(min(dst[0][0], dst[1][0]), dst[2][0]), dst[3][0])
    nh = max(max(max(dst[0][1], dst[1][1]), dst[2][1]), dst[3][1]) - \
         min(min(min(dst[0][1], dst[1][1]), dst[2][1]), dst[3][1])

    result = cv2.warpPerspective(img, warpR, (nw, nh))
    return result, warpR


def rad(x):
    return x * np.pi / 180


class preproc(object):
    def __init__(self, img_dim, rgb_means):
        self.img_dim = img_dim
        self.rgb_means = rgb_means

    def __call__(self, image, targets):
        assert len(targets) > 0, "this image does not have gt"
        height, width, _ = image.shape
        image_t = _resize_subtract_mean(image, self.img_dim, self.rgb_means)

        return image_t, targets