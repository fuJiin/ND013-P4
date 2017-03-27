from __future__ import absolute_import

import glob
import os

import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image, ImageDraw, ImageFont

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
    rgb = binary.rgb_select(undistort, thresh=(95, 190))
    hls = binary.hls_select(undistort, thresh=(90, 255))

    # Combine colors
    colors = np.zeros_like(rgb)
    colors[(hls == 1) & (rgb == 1)] = 1

    # Add gradients
    mag = binary.mag_thresh(undistort, thresh=(20, 255))
    direction = binary.dir_thresh(undistort, thresh=(0.6, 1.4), kernel=15)

    combined = np.zeros_like(rgb)
    combined[((colors == 1) & (direction == 1)) |
             ((direction == 1) & (mag == 1)) |
             ((colors == 1) & (mag == 1))] = 1

    return combined


class Lines(object):

    def __init__(self, diff_thresh=15, max_history=5):
        self.lines = []
        self.diff_thresh = diff_thresh
        self.max_history = max_history

        self.left_diffs = []
        self.right_diffs = []

    def add_lanes(self, left_x, right_x):
        """Add left/right x to lines"""
        self.lines.append([left_x, right_x])

        if len(self.lines) > self.max_history:
            self.lines = self.lines[1:]

    def diff_raw(self, left_x, right_x):
        """Calculate difference between new left/right x and existing"""
        left_diff, right_diff = (0, 0)

        for idx, (l, r) in enumerate(self.lines):
            gamma = 1 / (len(self.lines) - idx)
            left_diff += (gamma * np.mean(abs(np.array(left_x) - np.array(l))))
            right_diff += (gamma * np.mean(abs(np.array(right_x) - np.array(r))))

        left_diff /= len(self.lines)
        right_diff /= len(self.lines)

        self.left_diffs.append(left_diff)
        self.right_diffs.append(right_diff)

        return left_diff, right_diff

    def validate(self, left_x, right_x):
        """Validate whether left_x and right_x are valid to history"""
        if len(self.lines) == 0:
            return True, True

        left_diff, right_diff = self.diff_raw(left_x, right_x)
        print('{}, {}'.format(left_diff, right_diff))
        return (left_diff <= self.diff_thresh), (right_diff <= self.diff_thresh)


class ImageProcessor(object):
    """Wrapper to process individual frames"""

    def __init__(self, mtx, dist, src, dst):
        self.mtx = mtx
        self.dist = dist
        self.src = src
        self.dst = dst
        self.frame = 0
        self.lines = Lines()

        self.left_curve = 0
        self.right_curve = 0
        self.center_offset = None

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

        valid_left, valid_right = self.lines.validate(sw.left_x, sw.right_x)
        sw_left_curve, sw_right_curve = sw.curvature_ft()

        if len(self.lines.lines) > 0:
            new_left, new_right = self.lines.lines[-1]  # default to previous lines

            if valid_left:
                new_left = sw.left_x
                self.left_curve = sw_left_curve

            if valid_right:
                new_right = sw.right_x
                self.right_curve = sw_right_curve

            self.center_offset = histogram.center_offset(
                img=warped,
                left_x=new_left,
                right_x=new_right
            )
        else:
            new_left, new_right = sw.left_x, sw.right_x
            self.left_curve, self.right_curve = sw.curvature_ft()
            self.center_offset = sw.center_offset()

        self.lines.add_lanes(new_left, new_right)

        # Add frame
        self.frame += 1

        # Unwarp image
        left_x, right_x = self.lines.lines[-1]

        unwarped = transform.unwarp_lane(
            warped,
            orig_img=undistort,
            src=self.src, dst=self.dst,
            plot_y=sw.plot_y,
            left_x=left_x, right_x=right_x
        )

        # Add curvature, center offset
        out_img = Image.fromarray(unwarped)
        font = ImageFont.truetype('/Library/Fonts/Arial.ttf', 32)
        draw = ImageDraw.Draw(out_img)

        draw.text(
            (10, 10),
            'Left curvature: {} ft'.format(round(self.left_curve, 2)),
            font=font
        )
        draw.text(
            (10, 58),
            'Right curvature: {} ft'.format(round(self.right_curve, 2)),
            font=font
        )
        draw.text(
            (10, 106),
            'Center offset: {} ft'.format(round(self.center_offset, 2)),
            font=font
        )
        return np.asarray(out_img)


def process_video(video_path, mtx, dist):
    """Process video, given calibration matrix and distortion"""
    clip1 = VideoFileClip(video_path)

    processor = ImageProcessor(
        mtx=mtx, dist=dist,
        src=transform.SOURCE,
        dst=transform.DEST
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
