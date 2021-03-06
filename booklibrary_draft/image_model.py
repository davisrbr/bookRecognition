from typing import Callable, Dict, List, Tuple
import numpy as np
import torchvision
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import cv2

# from itertools import combinations


def image_model_pytorch(
    img_path: Path,
    model_str: str = "resnet50",
    threshold: float = 0.8,
    book_limit: int = 15,
) -> List[np.ndarray]:
    """Accepts an image path and returns
    np.ndarary images of books, using
    a PyTorch model."""
    processed_img = process_image(img_path)
    model = object_detect_pytorch(model_str)

    # make prediction
    prediction = model(processed_img)
    rects = prediction[0]["boxes"]
    img_scaled = processed_img.mul(255).byte()[0, ...]
    im = img_scaled.permute(1, 2, 0).numpy()

    # check threshold and slice books
    sliced_books: List[np.ndarray] = []
    for index, rect in enumerate(rects):
        if (
            prediction[0]["scores"][index] > threshold
            and prediction[0]["labels"][index] == 84
        ):
            xmin, ymin, xmax, ymax = [int(i) for i in rect]
            book = im[ymin:ymax, xmin:xmax, :]
            # rotate if tilted
            if (xmax - xmin) > (ymax - ymin):
                book = np.rot90(book)
            sliced_books.append(book)
        if index >= book_limit:
            return sliced_books
    return sliced_books


def object_detect_pytorch(
    model_str: str = "resnet50",
) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    if model_str == "resnet50":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    else:
        # not implemented
        raise NotImplementedError(f"Object detection model {model_str} not implemented")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model


def process_image(
    img_path: Path, resize: bool = False, normalize: bool = False
) -> np.ndarray:
    img = Image.open(img_path).convert("RGB")
    # define transformation
    process = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if normalize
            else lambda x: x,
            transforms.Resize((224, 224)) if resize else lambda x: x,
        ]
    )
    tensor = process(img)
    tensor_unsqueezed = tensor.unsqueeze(0)
    return tensor_unsqueezed


def deskew(image_rgb: np.ndarray) -> np.ndarray:
    """Attempts to deskew an rgb np.ndarray image.
    A part of a few methods to transform an image of a book to improve OCR.
    I found that OpenCV's warpAffine was much faster than scikit-image's.

    9.85 ms ± 391 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)"""
    if not image_rgb.ndim == 3:
        raise TypeError(
            "the deskew transformation should be performed on "
            + "numpy rgb images of books"
        )

    image_grey = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(image_grey > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image_grey.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_rgb,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def rotate90(image_rgb: np.ndarray, rotations: int) -> Tuple[np.ndarray, int]:
    """Rotates an image by 90 degrees, given a tuple of the image and
    an integer number of previously performed rotations."""
    rotated_img = np.rot90(image_rgb, rotations + 1)
    return (rotated_img, rotations + 1)


def screen_books(bookshelf: List[np.ndarray]) -> List[np.ndarray]:
    """Screens objects that are unlikely to be books to save time.
    For first implementation, it (minimally) removes
    the images that are too large.
    Could make it resample for the text model as well.
    Remember: screen, then deskew"""
    book_shapes = [book.shape for book in bookshelf]
    height, width, _ = zip(*book_shapes)
    w_mean = sum(width) / len(width)
    h_mean = sum(height) / len(height)
    w_var = sum((i - w_mean) ** 2 for i in width) / len(width)
    h_var = sum((i - h_mean) ** 2 for i in height) / len(height)

    # as a rough first pass, assume normally distributed and keep w/n 95%
    w_factor = w_mean + np.sqrt(w_var) * 2
    h_factor = h_mean + np.sqrt(h_var) * 2

    screened = [
        book
        for index, book in enumerate(bookshelf)
        if height[index] < h_factor and width[index] < w_factor
    ]

    return screened
