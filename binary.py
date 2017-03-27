import cv2
import numpy as np


def rgb_select(img, ch_filter=0, thresh=(0, 255)):
    """Select channels within given threshold in RGB space"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    s_channel = hls[:,:,ch_filter]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) &
                  (s_channel <= thresh[1])] = 1

    return binary_output


def hsv_select(img, ch_filter=2, thresh=(0, 255)):
    """Select channels within given threshold in HSV space"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hls[:,:,ch_filter]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) &
                  (s_channel <= thresh[1])] = 1

    return binary_output


def hls_select(img, ch_filter=2, thresh=(0, 255)):
    """Select channels within given threshold in HLS space"""
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,ch_filter]

    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) &
                  (s_channel <= thresh[1])] = 1

    return binary_output


def abs_sobel_thresh(img, orient='x', kernel=3, thresh=(0, 255)):
    """Calculate directional gradient and apply threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if orient == 'x':
        sobel = sobel_x(gray, kernel=kernel)
    elif orient == 'y':
        sobel = sobel_y(gray, kernel=kernel)
    else:
        raise Exception('Unknown orient for Sobel')

    # Scale absolute gradient
    abs_sobel = np.absolute(sobel)
    scaled_sobel = scale_sobel(abs_sobel)

    # Apply mask
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def mag_thresh(img, kernel=3, thresh=(0, 255)):
    """Calculate gradient magnitude and apply threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take gradient in x, y separately
    sx = sobel_x(gray, kernel=kernel)
    sy = sobel_y(gray, kernel=kernel)

    # Calculate & scale magnitude
    mag = np.sqrt(sx ** 2 + sy ** 2)
    scaled_sobel = scale_sobel(mag)

    # Apply mask
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) &
                  (scaled_sobel <= thresh[1])] = 1

    return binary_output


def dir_thresh(img, kernel=3, thresh=(0, np.pi/2)):
    """Calculate gradient direction and apply threshold"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Take gradient in x, y separately
    sx = sobel_x(gray, kernel=kernel)
    sy = sobel_y(gray, kernel=kernel)

    # Calculate & scale direction
    direction = np.arctan2(
        np.absolute(sy),
        np.absolute(sx)
    )

    # Apply mask
    binary_output = np.zeros_like(direction)
    binary_output[(direction >= thresh[0]) &
                  (direction <= thresh[1])] = 1

    return binary_output


def sobel_x(img, kernel=3):
    """Gradient in x"""
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel)


def sobel_y(img, kernel=3):
    """Gradient in y"""
    return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel)


def scale_sobel(gradients):
    """Normalize gradients"""
    return np.uint8(255 * gradients / np.max(gradients))
