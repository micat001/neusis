import numpy as np
import cv2
import struct
import numpy as np


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def draw_sonar_msg(msg, return_processed=True, trim=True, verbose=False):
    num_ranges = len(msg.ranges)
    num_angles = len(msg.azimuth_angles)
    data_size = msg.data_size

    if data_size == 1:
        data_type = np.uint8
    elif data_size == 4:
        data_type = np.uint32
    else:
        raise ValueError

    raw_intensities = np.frombuffer(msg.intensities, dtype=data_type)

    if verbose:
        print(f"Size: {len(raw_intensities)}")
        print(f"Ranges: {num_ranges}")
        print(f"Angles: {num_angles}")

    # Range minimum from oculus specs:
    rng_minimum = np.argmax(np.array(msg.ranges) > 0.1)
    vmax = 1.0

    # Log scaling modified from sonar_postprocessor_nodelet.cpp
    # Avoid log(0)
    if return_processed:
        new_intensites = raw_intensities.astype(np.float32) + 1e-6
        v = np.log(new_intensites) / np.log(np.iinfo(data_type).max)
        vmax = 1.0
        intensity_threshold = 0.74
        v = (v - intensity_threshold) / (vmax - intensity_threshold)
        v = np.clip(v, a_min=0.0, a_max=1.0)
        v = (np.iinfo(np.uint8).max * v).astype(np.uint8)
        sonar_image = v.reshape(num_ranges, num_angles)
        sonar_image = np.flipud(sonar_image)

    else:
        sonar_image = raw_intensities.reshape(num_ranges, num_angles)
        sonar_image = sonar_image / np.iinfo(data_type).max
        sonar_image = np.flipud(sonar_image)

    if trim:
        sonar_image = sonar_image[:-rng_minimum, :]

        if verbose:
            print("Original Shape: ", sonar_image.shape)
            print("New Shape: ", sonar_image.shape)

    return sonar_image, np.array(msg.ranges), np.array(msg.azimuth_angles)
