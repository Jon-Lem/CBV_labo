import cv2
import numpy as np

MONITOR_W = 38.5
ipd = 6.5

def stereo_gen(left_img, depth):
    # print("left_img shape:",left_img.shape)
    h, w , c = left_img.shape

    depth_min = depth.min()
    depth_max = depth.max()
    # Scale depth values from 0 to 1
    depth = (depth - depth_min) / (depth_max - depth_min)

    right = np.zeros_like(left_img)

    deviation_cm = ipd * 0.12
    deviation = deviation_cm * MONITOR_W * (w / 1920)

    # print("\ndeviation:", deviation)

    for row in range(h):
        for col in range(w):
            col_r = col - int((1 - depth[row][col] ** 2) * deviation)
            if col_r >= 0:
                right[row][col_r] = left_img[row][col]

    right_fix = np.array(right)
    gray = cv2.cvtColor(right_fix, cv2.COLOR_BGR2GRAY)

    # Fill up empty pixels with one of the surrounding pixels
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

