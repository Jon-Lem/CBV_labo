import numpy as np
import cv2
IPD = 6.5
MONITOR_W = 38.5

def write_depth(depth, bits=1, reverse=True):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1

    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = 0
    if not reverse:
        out = max_val - out

    if bits == 2:
        depth_map = out.astype("uint16")
    else:
        depth_map = out.astype("uint8")

    return depth_map

def generate_stereo(left_img, depth):
    h, w, c = left_img.shape

    depth_min = depth.min()
    depth_max = depth.max()
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left_img)

    deviation_cm = IPD * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)

    print("\ndeviation:", deviation)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col] ** 2) * deviation)
            # col_r = col - int((1 - depth[row][col]) * deviation)
            if col_r >= 0:
                right[row][col_r] = left_img[row][col]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)
    rows, cols = np.where(gray == 0)
    for row, col in zip(rows, cols):
        for offset in range(1, int(deviation)):
            r_offset = col + offset
            l_offset = col - offset
            if r_offset < w and not np.all(right_fix[row][r_offset] == 0):
                right_fix[row][col] = right_fix[row][r_offset]
                break
            if l_offset >= 0 and not np.all(right_fix[row][l_offset] == 0):
                right_fix[row][col] = right_fix[row][l_offset]
                break

    return right_fix

def overlap(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]

    # final image
    composite = np.zeros((height2, width2, 3), np.uint8)

    # iterate through "left" image, filling in red values of final image
    for i in range(height1):
        for j in range(width1):
            try:
                composite[i, j, 2] = im1[i, j, 2]
            except IndexError:
                pass

    # iterate through "right" image, filling in blue/green values of final image
    for i in range(height2):
        for j in range(width2):
            try:
                composite[i, j, 1] = im2[i, j, 1]
                composite[i, j, 0] = im2[i, j, 0]
            except IndexError:
                pass

    return composite