import cv2
import numpy as np


SOURCE = np.float32([
    (507, 500),
    (792, 500),
    (150, 720),
    (1200, 720)
])
DEST = np.float32([
    (150, 300),
    (1200, 300),
    (150, 720),
    (1200, 720)
])


def transform(img, src, dst):
    """
    Transform an img using source and destination pts
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])

    return cv2.warpPerspective(
        img, M, img_size,
        flags=cv2.INTER_LINEAR
    )


def unwarp_lane(warped_img, orig_img,
                src, dst,
                plot_y, left_x, right_x,
                lane_color=(0, 255, 0)):
    """Map lanes back to unwarped image"""
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast x & y pts into usable format for cv2.fillPoly()
    pts_left = np.array([
        np.transpose(np.vstack([left_x, plot_y]))
    ])
    pts_right = np.array([
        np.flipud(np.transpose(np.vstack([right_x, plot_y])))
    ])
    pts = np.hstack((pts_left, pts_right))

    # Draw lane onto warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), lane_color)

    # Project back into original image
    Minv = cv2.getPerspectiveTransform(dst, src)

    new_warp = cv2.warpPerspective(
        color_warp, Minv,
        (orig_img.shape[1], orig_img.shape[0])
    )
    return cv2.addWeighted(orig_img, 1, new_warp, 0.3, 0)
