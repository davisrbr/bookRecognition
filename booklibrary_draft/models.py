from typing import Callable, Dict, List, Tuple
import numpy as np
import torchvision
from torchvision import transforms
import torch
from pathlib import Path
from PIL import Image
import keras_ocr
import tensorflow as tf
import logging
from symspellpy import SymSpell, Verbosity, symspellpy
import pkg_resources


def image_model(
        img_path: Path,
        model_str: str,
        model: Callable[[np.ndarray], Dict],
        threshold: float = 0.5,
        ) -> List[np.ndarray]:
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
            xmin, ymin, xmax, ymax = ([int(i) for i in rects])
            book = im[ymin:ymax, xmin:xmax, :]
            # rotate if tilted
            if (xmax - xmin) > (ymax - ymin):
                book = np.rot90(book)
            sliced_books.append(book)
    return sliced_books


def object_detect_pytorch(model_str: str = 'resnet50'):
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


def text_model(
        books: List[np.ndarray],
        text_model_str: str = 'keras_ocr',
        scale: float = 2
        ) -> List[List[str]]:
    if not all(4 >= book.ndim > 1 for book in books):
        raise TypeError(
                "the text model should be passed " +
                "a list of np.ndarray images of books"
                )
    tf.get_logger().setLevel(logging.ERROR)
    if text_model_str == 'keras_ocr':
        pipeline = keras_ocr.pipeline.Pipeline(scale=scale)
    else:
        raise NotImplementedError(
                f'Model {text_model_str} not yet implemented'
                )
    try:
        predictions = [pipeline.recognize([book])[0] for book in books]
    except RuntimeError as e:
        raise RuntimeError(f'TensorFlow/Keras OCR model not working: {e}')
    try:
        proposals = prioritize_position(predictions)
    except NotImplementedError as e:
        print(e)
        proposals = [[pred[i][0] for i, _ in enumerate(pred)]
                     for pred in predictions if pred]
    return proposals


def prioritize_position(
        predictions: List[Tuple[str, np.ndarray]]
        ) -> List[List[str]]:
    '''This proposes a word order given the placement of the
        identified words in the image'''
    raise NotImplementedError('prioritization not yet implemented')


def clean_proposals(
        proposals: List[List[str]],
        grab_authors: Callable[[List[str]], List[Tuple[List[str]]]],
        spell_correct: Callable[[List[str]], List[str]]
        ) -> List[List[str]]:
    '''Cleans proposals generated by the text_model
        Note: Should probably grab names before spell check!'''
    title_proposal = grab_authors(proposals)
    title_proposal_corrected = spell_correct(title_proposal)
    return title_proposal_corrected


def grab_authors(words: List[List[str]]) -> List[Tuple[List[str]]]:
    '''Spacy's Named Entity Recognition is not great for this.
        Ill want to train a custom model / create a db for authors'''
    return words


def spell_correct(
        title_proposal: List[str],
        max_edit: int = 2
        ) -> List[str]:
    dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
    # TODO: still need to find optimal edit distances
    sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit, prefix_length=7
            )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    def spell_check(input_term: str) -> List[symspellpy.SuggestItem]:
        return sym_spell.lookup(input_term,
                                Verbosity.CLOSEST,
                                max_edit_distance=max_edit,
                                include_unknown=False)

    proposals = [spell_check(word) for word in title_proposal]
    # I acknowledge this is too much and I should use a list comp--
    # there's definitely a straightforward way to not repeat spell_check,
    # but this is the best I can think of at the moment
    corrected_title = list(
            map(lambda term: term[0]._term, (filter(None, proposals)))
            )
    return corrected_title