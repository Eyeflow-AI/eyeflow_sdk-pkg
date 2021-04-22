"""
SiliconLife Eyeflow
Image manipulation functions

Author: Alex Sobral de Freitas

This code may contain code from:
    Copyright 2017-2018 Fizyr (https://fizyr.com)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import numpy as np
import cv2
from PIL import ImageFont
# ---------------------------------------------------------------------------------------------------------------------------------


def preprocess_image(image, mode):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        image: np.array of shape (None, None, 1 or 3)
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    image = image.astype(np.float32)

    if mode == 'tf':
        image /= 127.5
        image -= 1.
    elif mode == 'caffe':
        if image.ndim == 2:
            image = np.expand_dims(image, axis=-1)
            image -= 127.0
        elif image.shape[2] == 1:
            image -= 127.0
        elif image.shape[2] == 3:
            image[..., 0] -= 103.939
            image[..., 1] -= 116.779
            image[..., 2] -= 123.68

    return image
# ---------------------------------------------------------------------------------------------------------------------------------


def resize_image(image, target_height, target_width):
    """ Resize an image. Don't preserve image aspect ratio.

    Args
        target_height: Target height to resize
        target_width: Target width to resize

    Returns
        A resized image.
    """

    image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    return image
# ---------------------------------------------------------------------------------------------------------------------------------


def resize_image_scale(image, max_side):
    """ Resize an image such that the larger side equals max_side.
        Preserve image aspect ratio.

    Args
        max_side: The image's max side will be equal to max_side after resizing.

    Returns
        A resized image.
    """

    scale = max_side / max(image.shape[:2])

    # resize the image with the computed scale
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    return image, scale
# ---------------------------------------------------------------------------------------------------------------------------------


def resize_image_pad(image, target_height, target_width):
    """ Resize an image with padding to mantain aspect ratio.

    Args
        target_height: Target height to resize
        target_width: Target width to resize

    Returns
        A resized image with the target resolution and same aspect ratio.
    """
    (rows, cols) = (image.shape[0], image.shape[1])
    target_image = np.zeros((target_height, target_width, image.shape[2]), dtype=np.float)
    scale = min(target_width / cols, target_height / rows)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    target_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image

    return target_image, scale
# ---------------------------------------------------------------------------------------------------------------------------------


def convert_to_3_channels(image):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
    return image
#----------------------------------------------------------------------------------------------------------------------------------


def convert_to_1_channel(image):
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)
    elif image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.ndim == 2:
                image = np.expand_dims(image, axis=-1)

    return image
#----------------------------------------------------------------------------------------------------------------------------------


def convert_data_type(image, to='uint8'):
    '''
    Converts images represented as Numpy arrays between `uint8` and `float32`.
    Serves as a helper for certain photometric distortions. This is just a wrapper
    around `np.ndarray.astype()`.
    Arguments:
        to (string, optional): To which datatype to convert the input images.
            Can be either of 'uint8' and 'float32'.
    '''
    if not (to == 'uint8' or to == 'float32'):
        raise ValueError("`to` can be either of 'uint8' or 'float32'.")

    if to == 'uint8':
        image = np.round(image, decimals=0).astype(np.uint8)
    else:
        image = image.astype(np.float32)

    return image
#----------------------------------------------------------------------------------------------------------------------------------


def save_images_batch(images, image_path):

    image = np.squeeze(merge_images(images)).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 3:
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        cv2.imwrite(image_path, image)
#----------------------------------------------------------------------------------------------------------------------------------


def merge_images(images):
    max_shape = tuple(max(image.shape[x] for image in images) for x in range(3))

    size_w = int(np.ceil(np.sqrt(len(images))))
    size_h = int(np.ceil(len(images) / size_w))
    h, w = max_shape[0], max_shape[1]
    image_merged = np.zeros((max_shape[0] * size_h, max_shape[1] * size_w, max_shape[2]))

    for idx, image in enumerate(images):
        i = idx % size_w
        j = idx // size_w
        image_merged[j * h:j * h + image.shape[0], i * w:i * w + image.shape[1], :image.shape[2]] = image

    return image_merged
#----------------------------------------------------------------------------------------------------------------------------------


def get_draw_font(draw_obj, text, max_height=None, max_width=None):
    ini_size = 8

    try:
        font_name = './fonts/DejaVuSansMono.ttf'
        draw_font = ImageFont.truetype(font_name, ini_size)
    except:
        font_name = 'arial.ttf'
        draw_font = ImageFont.truetype(font_name, ini_size)

    if max_height is not None:
        end_size = max_height
        dim = 1
    elif max_width is not None:
        end_size = max_width
        dim = 0
    else:
        raise Exception("Must define max_height or max_width")

    txt_size = draw_obj.textsize(text, font=draw_font)
    while txt_size[dim] < end_size:
        ini_size += 1
        draw_font = ImageFont.truetype(font_name, ini_size)
        txt_size = draw_obj.textsize(text, font=draw_font)

    return draw_font
#----------------------------------------------------------------------------------------------------------------------------------
