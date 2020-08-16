import cv2
import matplotlib.pyplot as plt
import os
from typing import Callable
import numpy as np


def plot_ocr(filteredbooks: str,
             readtext: Callable[[str], str],
             blur: bool = False,
             morph: bool = False,
             gray: bool = False,
             deskew: bool = False) -> (plt.figure, plt.axes):
    figure, axes = plt.subplots(nrows=8, ncols=3, figsize=(30, 30))
    col, row = 0, 0
    for index in range(24):
        # read image
        image = cv2.imread(os.path.join(filteredbooks, f'{row}_{col}.jpg'))
        # pre-process
        if blur:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if morph:
            image = cv2.morphologyEx(image,
                                     cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8)
                                     )
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if deskew:
            temp = np.asarray(image)
            image = deskew_image(temp)

        # plot final
        axes[row, col].imshow(image, cmap="gray")
        axes[row, col].axis('off')
        plt.axis('off')

        # OCR text extraction
        output = readtext(image)
        data = ''
        for _, val in enumerate(output):
            data += val[1]+' '
        axes[row, col].set_title(data)

        # column/row logic for sub-plots
        if col == 2:
            col = 0
            row += 1
        else:
            col += 1

    # delete the final two empty axes
    figure.delaxes(axes.flatten()[-1])
    figure.delaxes(axes.flatten()[-2])
    return figure, axes


# I found that OpenCV's warpAffine was much faster than scikit-image's
def deskew_image(image: np.ndarray) -> np.ndarray:
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image,
                             matrix,
                             (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated
