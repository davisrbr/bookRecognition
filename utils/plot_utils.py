import cv2
import matplotlib.pyplot as plt
import os
from typing import Callable, Dict, List
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


def drawAnnotations(
        image: np.ndarray, predictions: Dict[str, np.ndarray], ax=None
        ) -> np.ndarray:
    """Draw text annotations onto image.
    Args:
        image: The image on which to draw
        predictions: The predictions as provided by `pipeline.recognize`.
        ax: A matplotlib axis on which to draw.
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.imshow(
            drawBoxes(
                image=image, boxes=predictions, boxes_format='predictions'
                )
            )
    predictions = sorted(predictions, key=lambda p: p[1][:, 1].min())
    left = []
    right = []
    for word, box in predictions:
        if box[:, 0].min() < image.shape[1] / 2:
            left.append((word, box))
        else:
            right.append((word, box))
    ax.set_yticks([])
    ax.set_xticks([])
    for side, group in zip(['left', 'right'], [left, right]):
        for index, (text, box) in enumerate(group):
            y = 1 - (index / len(group))
            xy = box[0] / np.array([image.shape[1], image.shape[0]])
            xy[1] = 1 - xy[1]
            ax.annotate(s=text,
                        xy=xy,
                        xytext=(-0.05 if side == 'left' else 1.05, y),
                        xycoords='axes fraction',
                        arrowprops={
                            'arrowstyle': '->',
                            'color': 'r'
                        },
                        color='r',
                        fontsize=12,
                        horizontalalignment='right' if side == 'left' else 'left'
                        )
    return ax


def drawBoxes(
        image: np.ndarray,
        boxes: List[int],
        color=(255, 0, 0),
        thickness=3,
        boxes_format='boxes'
        ):
    """Draw boxes onto an image.
    Args:
        image: The image on which to draw the boxes.
        boxes: The boxes to draw.
        color: The color for each box.
        thickness: The thickness for each box.
        boxes_format: The format used for providing the boxes. Options are
            "boxes" which indicates an array with shape(N, 4, 2) where N is the
            number of boxes and each box is a list of four points) as provided
            by `keras_ocr.detection.Detector.detect`, "lines" (a list of
            lines where each line itself is a list of (box, character) tuples)
            as provided by `keras_ocr.data_generation.get_image_generator`,
            or "predictions" where boxes is by itself a list of (word, box)
            tuples as provided by `keras_ocr.pipeline.Pipeline.recognize` or
            `keras_ocr.recognition.Recognizer.recognize_from_boxes`.
    """
    if len(boxes) == 0:
        return image
    canvas = image.copy()
    if boxes_format == 'lines':
        revised_boxes = []
        for line in boxes:
            for box, _ in line:
                revised_boxes.append(box)
        boxes = revised_boxes
    if boxes_format == 'predictions':
        revised_boxes = []
        for _, box in boxes:
            revised_boxes.append(box)
        boxes = revised_boxes
    for box in boxes:
        cv2.polylines(img=canvas,
                      pts=box[np.newaxis].astype('int32'),
                      color=color,
                      thickness=thickness,
                      isClosed=True)
    return canvas
