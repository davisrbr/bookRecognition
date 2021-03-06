from typing import List, Tuple, Callable

import numpy as np
import keras_ocr
import tensorflow as tf
from symspellpy import SymSpell, Verbosity, symspellpy
import pkg_resources
from image_model import rotate90, screen_books

import logging
import urllib.request
import json

from tqdm import tqdm
import matplotlib.pyplot as plt


KerasModel = Callable[[np.ndarray], List[List[Tuple[str, np.ndarray]]]]


def keras_model(text_model_str: str = "keras_ocr", scale: float = 2) -> KerasModel:
    tf.get_logger().setLevel(logging.ERROR)
    if text_model_str == "keras_ocr":
        pipeline = keras_ocr.pipeline.Pipeline(scale=scale)
    else:
        raise NotImplementedError(f"Model {text_model_str} not yet implemented")
    return pipeline.recognize


def text_model_inference(
    books: List[np.ndarray], model: KerasModel = keras_model()
) -> List[List[str]]:
    if not all(4 >= book.ndim > 1 for book in books):
        raise TypeError(
            "the text model should be passed " + "a list of np.ndarray images of books"
        )
    try:
        if len(books) > 1:
            print("Performing text prediction on books")
        predictions = []
        for _, book in enumerate(tqdm(books, disable=len(books) == 1)):
            predictions.append(model([book])[0])
        # predictions = [pipeline.recognize([book])[0] for book in books]
    except RuntimeError as e:
        raise RuntimeError(f"Keras OCR model not working: {e}")
    try:
        proposals = prioritize_position(predictions)
    except NotImplementedError:
        proposals = [
            [pred[i][0] for i, _ in enumerate(pred)] for pred in predictions if pred
        ]
    return proposals


# TODO: Define in beginning of set-up/call
# i.e. need to make a models set up for image, text, spell
#############################################
max_edit = 2
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
# TODO: still need to find optimal edit distances
sym_spell = SymSpell(max_dictionary_edit_distance=max_edit, prefix_length=7)
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

SymSpellType = Callable[[str], List[symspellpy.SuggestItem]]


def spell_check(input_term: str) -> List[symspellpy.SuggestItem]:
    return sym_spell.lookup(
        input_term, Verbosity.CLOSEST, max_edit_distance=max_edit, include_unknown=False
    )


#############################################


def grab_authors(words: List[List[str]]) -> List[Tuple[List[str]]]:
    """Spacy's Named Entity Recognition is not great for this.
    Ill want to train a custom model / create a db for authors"""
    return words


def spell_correct(
    title_proposal: List[str],
    max_edit: int = 2,
    spell_check: SymSpellType = spell_check,
) -> List[str]:
    # spell check and remove most single letters
    proposals = [
        spell_check(word)
        for word in title_proposal
        if (len(word) > 1 or word in {"i", "I", "a", "A"})
    ]
    # this is too confusing and I should replace with a list comprehension
    # corrected_title = [proposal[0]._term for proposal in proposals
    #                    if proposal is None]
    corrected_title = list(map(lambda term: term[0]._term, (filter(None, proposals))))
    return corrected_title


def prioritize_position(predictions: List[Tuple[str, np.ndarray]]) -> List[List[str]]:
    """This proposes a word order given the placement of the
    identified words in the image"""
    raise NotImplementedError("Robust prioritization not yet implemented")


AuthorSlicer = Callable[[List[str]], List[Tuple[List[str]]]]
SpellCorrect = Callable[[List[str]], List[str]]


def clean_proposals(
    proposals: List[List[str]],
    grab_authors: AuthorSlicer = grab_authors,
    spell_correct: SpellCorrect = spell_correct,
) -> List[List[str]]:
    """Cleans proposals generated by the text_model
    Note: Should probably grab names before spell check!"""
    cleaned_proposals = []
    for proposal in proposals:
        title_proposal = grab_authors(proposal)
        title_proposal_corrected = spell_correct(title_proposal)
        cleaned_proposals.append(title_proposal_corrected)
    return cleaned_proposals


def format_words_open_library(words: List[str]) -> str:
    """Format title words for the open library api"""
    try:
        sight = "http://openlibrary.org/search.json?title=" + "+".join(words)
    # likely empty title
    except TypeError:
        sight = ""
    return sight


def get_book_open_library(url: str, option: int = 0) -> str:
    """Retrieve book from the open library api"""
    if url == "":
        return ""
    try:
        with urllib.request.urlopen(url) as url:
            data = json.loads(url.read().decode())
        try:
            title = data["docs"][option]["title_suggest"]
        except IndexError:
            title = ""
    except urllib.error.HTTPError:
        title = ""
    return title


def generate_title_possibilities(
    word_list: List[str], rolling_combinations: bool = False
) -> List[List[str]]:
    """generates possible title once a first attempt fails. first generated:
        - remove leading one and two words
        - remove trailing one and two words
        - removing leading, trailing; combinations of one and two
    (where we use that most authors have 1 or 2 names on book cover)
    the remaining generated titles may exhaustively do unordered
    samples without replacement.

    Note: all attempts use a minimum of three words.
    Other note: this schema works a lot better when words are
                reasonably ordered."""
    # word_list = title_words.split(' ')
    if len(word_list) <= 2:
        return word_list

    if rolling_combinations and 6 < len(word_list) < 12:
        possible1 = [word_list[i : i + 5] for i in range(len(word_list) - 5)]
        possible2 = [word_list[i : i + 3] for i in range(len(word_list) - 3)]
        return possible1 + possible2

    # slice first two and last two words, ordered by priority
    possible = [
        word_list[1:],
        word_list[:-1],
        word_list[2:],
        word_list[:-2],
        word_list[1:-1],
    ]
    if len(word_list) > 4:
        possible.append(word_list[2:-1])
        possible.append(word_list[1:-2])

    return possible


def title_metric(title_proposal: List[str]) -> bool:
    """Simple metric to indicate if a proposed title
    is valid or worth searching."""
    word_lengths = [len(word) for word in title_proposal]
    # this will fail on these books, so author NER is necessary
    # https://www.abebooks.com/books/single-letter-title-shortest-mccarthy/warhol-updike.shtml
    try:
        first_condition = max(word_lengths) > 4
    except ValueError:
        # word_lengths is an empty list
        return False
    # these heuristics are best guesses, should check more empirically
    second_condition = len(word_lengths) > 3 and max(word_lengths) > 3
    third_condition = sum(word_lengths) / len(word_lengths) >= 4.0
    return first_condition or second_condition or third_condition


def title_metric_compare(title_proposal: List[str]) -> int:
    """Comparitive metric to indicate the strength of a
    proposed title."""
    word_lengths = [len(word) for word in title_proposal]
    try:
        metric = max(word_lengths) * len(word_lengths)
    except ValueError:
        metric = len(word_lengths)
    return metric


def books_from_proposed(
    books: List[np.ndarray], display=False, verbose=False
) -> List[str]:
    """Orchestrates taking books (produced by the image model),
    cleaning and generating title possibilities, and
    searching for them."""
    # remove images that likely are not books
    screened_books: List[np.ndarray] = screen_books(books)
    # assuming books have been e.g. deskewed
    proposed_titles: List[List[str]] = text_model_inference(screened_books)
    # perform initial title clean, assuming in ENGLISH
    cleaned_titles: List[List[str]] = clean_proposals(proposed_titles)

    relevant_titles = []
    for index, title in enumerate(cleaned_titles):
        if title_metric(title):
            relevant_titles.append(title)
            if verbose:
                print(f"title: {title}")

            if verbose:
                print(f"cleaned title: {cleaned_titles[index]}")
            if display:
                plt.imshow(screened_books[index])
                plt.show()

            title_score = title_metric_compare(title)
            rotation = 0
            # check goodness of title and number of rotations already performed
            while title_score < 50 and rotation < 3:
                # rotates and rotation += 1
                rotated_book, rotation = rotate90(screened_books[index], rotation)

                if display:
                    plt.imshow(rotated_book)
                    plt.show()

                # test new rotation
                rotated_title: List[List[str]] = text_model_inference([rotated_book])
                cleaned_rotated: List[str] = clean_proposals(rotated_title)[0]
                if verbose:
                    print(f"cleaned rotated title: {cleaned_rotated}")
                new_metric = title_metric_compare(cleaned_rotated)
                # check improvement, update title/metric if improved
                if new_metric > title_score:
                    relevant_titles.append(cleaned_rotated)
                    title_score = new_metric
        else:
            relevant_titles.append([])

    # pull from open library api
    print("Pulling from open library api:")
    found_books = []
    for _, title in enumerate(tqdm(relevant_titles)):
        if verbose:
            print(title)
        # make a first try
        if title:
            format_title = format_words_open_library(title)
            found_title = get_book_open_library(format_title)
        else:
            found_title = ""
        # empty string if not found (either from get_book method or empty list)
        if found_title:
            found_books.append(found_title)
        # otherwise generate new possibilities
        else:
            if verbose:
                print(f"alt title for: {title}")
            alt_titles = generate_title_possibilities(title)
            for _, alt_title in enumerate(alt_titles):
                # should add a compare to title_metric_compare(alt_title)
                if title_metric(alt_title):
                    format_alt_title = format_words_open_library(alt_title)
                    found_alt_title = get_book_open_library(format_alt_title)
                    if found_alt_title:
                        found_books.append(found_alt_title)
                        break
    return found_books
