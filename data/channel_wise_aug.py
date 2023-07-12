import random

import PIL
import cv2
import numpy as np
from imgaug import augmenters as iaa
from scipy import ndimage
from vidaug import augmentors as va

random.seed(42)


def identity(images):
    return images


def eraser_gray(images, s_l=0.1, s_h=0.2, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    input_img = images[0]
    img_h, img_w = input_img.shape

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w))

    else:
        c = np.random.uniform(v_l, v_h)

    cutOuts = []
    for idx, input_img in enumerate(images):
        input_img[top:top + h, left:left + w] = c
        cutOuts.append(input_img)
    return np.array(cutOuts)


def eraser(images, s_l=0.1, s_h=0.2, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255, pixel_level=False):
    input_img = images[0]
    inp_shape = input_img.shape

    if len(inp_shape) == 3:
        img_h, img_w, img_c = inp_shape
    else:
        img_h, img_w = inp_shape
        img_c = 1

    while True:
        s = np.random.uniform(s_l, s_h) * img_h * img_w
        r = np.random.uniform(r_1, r_2)
        w = int(np.sqrt(s / r))
        h = int(np.sqrt(s * r))
        left = np.random.randint(0, img_w)
        top = np.random.randint(0, img_h)

        if left + w <= img_w and top + h <= img_h:
            break

    if pixel_level:
        c = np.random.uniform(v_l, v_h, (h, w, img_c))

    else:
        c = np.random.uniform(v_l, v_h)

    cutOuts = []
    for idx, input_img in enumerate(images):
        input_img[top:top + h, left:left + w] = c
        cutOuts.append(input_img)
    return np.array(cutOuts)


def translation(images):
    transformed_imgs = []
    T = np.float32([[1, 0, 10], [0, 1, 10]])

    input_img = images[0]
    inp_shape = input_img.shape

    if len(inp_shape) == 3:
        height, width, channel = inp_shape
    else:
        height, width = inp_shape

    for image in images:
        transformed_img = cv2.warpAffine(image, T, (width, height))
        transformed_imgs.append(transformed_img)

    return np.array(transformed_imgs)


def blur(images, k=5):
    aug = iaa.MotionBlur(k=k, angle=[-45, 45])
    blurred_images = aug(images=images)
    return np.array(blurred_images)


def brightness(images):  # [0.1,1.9]
    v = random.uniform(0.1, 1.9)
    assert 0.1 <= v <= 1.9
    return np.array([np.array(PIL.ImageEnhance.Brightness(PIL.Image.fromarray(img)).enhance(v)) for img in images])


def salt(images):
    seq = va.Sequential([va.Salt(ratio=20)])
    video_aug = seq(images)
    return np.array(video_aug)


def pepper(images):
    seq = va.Sequential([va.Pepper(ratio=20)])
    video_aug = seq(images)
    return np.array(video_aug)


def consistent_brightness_i(images):
    value = 20.0
    brightness_imgs = []
    for image in images:
        brightness_img = image + value
        brightness_imgs.append(brightness_img)
    return np.array(brightness_imgs)


def consistent_brightness_d(images):
    value = -20.0
    brightness_imgs = []
    for image in images:
        brightness_img = image + value
        brightness_imgs.append(brightness_img)
    return np.array(brightness_imgs)


def rotate(images):
    value = 10
    rotate_imgs = []
    for image in images:
        rotate_img = ndimage.rotate(image, value)
        rotate_imgs.append(rotate_img)
    return np.array(rotate_imgs)


augmentations = {'identity': identity,
                 'brightness': brightness,
                 'blur': blur,
                 'salt': salt,
                 'pepper': pepper,
                 'rotate': rotate,
                 'eraser': eraser}
available_augmentations = [
    'identity',
    'brightness',
    'blur',
    'salt',
    'pepper',
    'rotate',
    'eraser']
