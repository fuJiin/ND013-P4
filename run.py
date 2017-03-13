from __future__ import absolute_import

import glob
import os

import numpy as np
from moviepy.editor import VideoFileClip

import binary
import calibrate
import cv2
import transform
from lanes import histogram


def calibrate_camera():
    """
    Calibrate camera on calibration images

    :return mtx: calibration matrix
    :return dist: distortion
    """

    calib_paths = glob.glob('./camera_cal/*.jpg')
    obj_pts, img_pts = calibrate.collect_obj_img_pts(calib_paths)

    test_image = cv2.imread(calib_paths[0])
    img_size = (test_image.shape[1], test_image.shape[0])

    _, mtx, dist, _, _ = calibrate.calibrate_camera(
        obj_pts, img_pts, img_size)

    return mtx, dist


def apply_filters(undistort):
    """Apply HLS and Sobel filters"""
    hls = binary.hls_select(undistort, thresh=(90, 255))
    mag = binary.mag_thresh(undistort, thresh=(20, 255))
    direction = binary.dir_thresh(undistort, thresh=(0.6, 1.4), kernel=15)

    combined = np.zeros_like(hls)
    combined[((hls == 1) & (direction == 1)) |
             ((direction == 1) & (mag == 1)) |
             ((hls == 1) & (mag == 1))] = 1

    return combined


class ImageProcessor(object):
    """Wrapper to process individual frames"""

    def __init__(self, mtx, dist, src, dst):
        self.mtx = mtx
        self.dist = dist
        self.src = src
        self.dst = dst
        self.frame = 0

    def process_image(self, img):
        print('> Processing frame {}'.format(self.frame))
        undistort = calibrate.undistort(img, self.mtx, self.dist)
        binary = apply_filters(undistort)

        # Warp binary image
        warped = transform.transform(
            binary,
            src=self.src, dst=self.dst
        )
        # Search for lane
        sw = histogram.SlidingWindow(warped)
        sw.search()

        # Draw new image
        self.frame += 1

        return transform.unwarp_lane(
            warped,
            orig_img=undistort,
            src=self.src, dst=self.dst,
            plot_y=sw.plot_y,
            left_x=sw.left_x, right_x=sw.right_x
        )


def process_video(video_path, mtx, dist):
    """Process video, given calibration matrix and distortion"""
    clip1 = VideoFileClip(video_path)

    processor = ImageProcessor(
        mtx=mtx, dist=dist,
        src=transform.SOURCE,
        dst=transform.DEST,
        line=Line()
    )
    clip = clip1.fl_image(processor.process_image)

    video_name = os.path.basename(video_path)
    write_path = './processed_{}'.format(video_name)
    clip.write_videofile(write_path, audio=False)


if __name__ == '__main__':
    print('Calibrating camera...')
    mtx, dist = calibrate_camera()

    print('Processing video...')
    process_video('./project_video.mp4', mtx=mtx, dist=dist)
