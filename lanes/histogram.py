import numpy as np
import cv2
import matplotlib.pyplot as plt


class SlidingWindow(object):
    """Sliding window search to find lanes"""

    def __init__(self, img,
                 windows=9, margin=100, min_pix=50,
                 prev_left_fit=None, prev_right_fit=None,
                 collector=None):
        self.img = img
        self.windows = windows
        self.margin = margin
        self.min_pix = min_pix

        self.prev_left_fit = prev_left_fit
        self.prev_right_fit = prev_right_fit

        self.non_zero_x, self.non_zero_y = self._extract_non_zeroes(img)

    def init_search(self):
        """
        Find left and right lane pixel indices
        from scratch using sliding windows
        """
        hgram = histogram(self.img)

        mid_pt = np.int(hgram.shape[0] / 2)
        left_base = np.argmax(hgram[:mid_pt])
        right_base = np.argmax(hgram[mid_pt:]) + mid_pt

        window_height = np.int(self.img.shape[0] / self.windows)

        left_curr = left_base
        right_curr = right_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.windows):
            win_y_low = self.img.shape[0] - (window + 1) * window_height
            win_y_high = self.img.shape[0] - (window * window_height)

            win_xleft_low = left_curr - self.margin
            win_xleft_high = left_curr + self.margin

            win_xright_low = right_curr - self.margin
            win_xright_high = right_curr + self.margin

            good_left_inds = (
                (self.non_zero_y >= win_y_low) &
                (self.non_zero_y < win_y_high) &
                (self.non_zero_x >= win_xleft_low) &
                (self.non_zero_x < win_xleft_high)
            ).nonzero()[0]

            good_right_inds = (
                (self.non_zero_y >= win_y_low) &
                (self.non_zero_y < win_y_high) &
                (self.non_zero_x >= win_xright_low) &
                (self.non_zero_x < win_xright_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.min_pix:
                left_curr = np.int(np.mean(self.non_zero_x[good_left_inds]))

            if len(good_right_inds) > self.min_pix:
                right_curr = np.int(np.mean(self.non_zero_x[good_right_inds]))

        return np.concatenate(left_lane_inds), np.concatenate(right_lane_inds)

    def search(self):
        """Find lanes, using previous positions if available"""
        if self.prev_left_fit and self.prev_right_fit:
            left_lane_inds = self.extract_lane_inds(self.prev_left_fit)
            right_lane_inds = self.extract_lane_inds(self.right_left_fit)
        else:
            left_lane_inds, right_lane_inds = self.init_search()

        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        self.left_fit = self.fit_line(left_lane_inds)
        self.right_fit = self.fit_line(right_lane_inds)

        self.plot_y = np.linspace(0, self.img.shape[0] - 1, self.img.shape[1])

        self.left_x = (
            self.left_fit[0] * self.plot_y ** 2 +
            self.left_fit[1] * self.plot_y +
            self.left_fit[2]
        )
        self.right_x = (
            self.right_fit[0] * self.plot_y ** 2 +
            self.right_fit[1] * self.plot_y +
            self.right_fit[2]
        )

    def extract_lane_inds(self, prev_fit):
        """Extract lane indices from previous fit"""
        center = (
            prev_fit[0] * (non_zero_y ** 2) +
            prev_fit[1] * non_zero_y +
            prev_fit[2] - self.margin
        )
        return (
            (self.non_zero_x > (center - self.margin)) &
            (self.non_zero_x < (center + self.margin))
        )

    def fit_line(self, lane_inds, fit_order=2):
        """Fit polynomial line to x and y points"""
        x = self.non_zero_x[lane_inds]
        y = self.non_zero_y[lane_inds]
        return np.polyfit(y, x, fit_order)

    def visualize(self,
                  line_color='yellow',
                  left_color=(255, 0, 0),
                  right_color=(0, 0, 255),
                  margin_color=(0, 255, 0)):
        """Output image to visualize results"""
        out_img = np.dstack((self.img, self.img, self.img)) * 255
        window_img = np.zeros_like(out_img)

        # Color left and right lines
        out_img[
            self.non_zero_y[self.left_lane_inds],
            self.non_zero_x[self.left_lane_inds]
        ] = left_color

        out_img[
            self.non_zero_y[self.right_lane_inds],
            self.non_zero_x[self.right_lane_inds]
        ] = right_color

        # Generate polygon to illustrate search window areas
        # and recast pts into usable format for cv2.fillPoly()
        plot_y = np.linspace(0, self.img.shape[0] - 1, self.img.shape[1])

        # Left side
        left_line_window1 = np.array([
            np.transpose(
                np.vstack([self.left_x - self.margin, self.plot_y])
            )
        ])
        left_line_window2 = np.array([
            np.flipud(
                np.transpose(
                    np.vstack([self.left_x + self.margin, self.plot_y])
                )
            )

        ])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))

        # Right side
        right_line_window1 = np.array([
            np.transpose(
                np.vstack([self.right_x - self.margin, self.plot_y])
            )
        ])
        right_line_window2 = np.array([
            np.flipud(
                np.transpose(
                    np.vstack([self.right_x + self.margin, self.plot_y])
                )
            )
        ])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw lane
        cv2.fillPoly(window_img, np.int_([left_line_pts]), margin_color)
        cv2.fillPoly(window_img, np.int_([right_line_pts]), margin_color)

        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        plt.imshow(result)
        plt.plot(self.left_x, self.plot_y, color=line_color)
        plt.plot(self.right_x, self.plot_y, color=line_color)
        plt.xlim(0, self.img.shape[1])
        plt.ylim(self.img.shape[0], 0)

    def curvature_ft(self, y_ft_per_px=10/200, x_ft_per_px=12/800):
        """Curvature in meters"""
        y_eval = np.max(self.plot_y)

        left_fit_cr = np.polyfit(
            self.plot_y * y_ft_per_px,
            self.left_x * x_ft_per_px, 2
        )
        right_fit_cr = np.polyfit(
            self.plot_y * y_ft_per_px,
            self.right_x * x_ft_per_px, 2
        )
        # Calculate the new radii of curvature
        left_rad = ((1 + (2 * left_fit_cr[0] * y_eval * y_ft_per_px +
                          left_fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * left_fit_cr[0])
        right_rad = ((1 + (2 * right_fit_cr[0] * y_eval * y_ft_per_px +
                           right_fit_cr[1]) ** 2) ** 1.5) / \
                    np.absolute(2 * right_fit_cr[0])

        return left_rad, right_rad

    def center_offset(self, x_ft_per_px=12/800):
        """Find x offset of car in relation to the lane"""
        frame_center = self.img.shape[1] / 2
        lane_center = (self.right_x[-1] - self.left_x[-1]) / 2
        return (frame_center - lane_center) * x_ft_per_px

    def _extract_non_zeroes(self, img):
        """Extract non_zero_x, non_zero_y from image"""
        non_zero = img.nonzero()
        return np.array(non_zero[1]), np.array(non_zero[0])


def histogram(img):
    """Histogram of pixels in an image"""
    return np.sum(img[int(img.shape[0]/2):,:], axis=0)
