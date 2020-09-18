from typing import Callable, Dict, List
import numpy as np
import torchvision
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
# from itertools import combinations


def image_model_pytorch(
        img_path: Path,
        model_str: str = 'resnet50',
        threshold: float = 0.8,
        book_limit: int = 2,
        ) -> List[np.ndarray]:
    '''Accepts an image path and returns
        np.ndarary images of books, using
        a PyTorch model.'''
    processed_img = process_image(img_path)
    model = object_detect_pytorch(model_str)

    # make prediction
    prediction = model(processed_img)
    rects = prediction[0]['boxes']
    img_scaled = processed_img.mul(255).byte()[0, ...]
    im = img_scaled.permute(1, 2, 0).numpy()

    # check threshold and slice books
    sliced_books: List[np.ndarray] = []
    for index, rect in enumerate(rects):
        if (prediction[0]['scores'][index] > threshold and
                prediction[0]['labels'][index] == 84):
            xmin, ymin, xmax, ymax = ([int(i) for i in rect])
            book = im[ymin:ymax, xmin:xmax, :]
            # rotate if tilted
            if (xmax - xmin) > (ymax - ymin):
                book = np.rot90(book)
            sliced_books.append(book)
        if index >= book_limit:
            return sliced_books
    return sliced_books


def object_detect_pytorch(model_str: str = 'resnet50'
                          ) -> Callable[[np.ndarray], Dict[str, np.ndarray]]:
    if model_str == 'resnet50':
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True
                )
    else:
        # not implemented
        raise NotImplementedError(
                f'Object detection model {model_str} not implemented'
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model


def process_image(
        img_path: Path,
        resize: bool = False,
        normalize: bool = False
        ) -> np.ndarray:
    img = Image.open(img_path).convert('RGB')
    # define transformation
    process = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]) if normalize else lambda x: x,
        transforms.Resize(
            (224, 224)) if resize else lambda x: x
        ])
    tensor = process(img)
    tensor_unsqueezed = tensor.unsqueeze(0)
    return tensor_unsqueezed
