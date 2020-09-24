from pathlib import Path
from image_model import image_model_pytorch


books = Path('/Users/davisbrown/Desktop/SideProjects/bookRecognition/bookshelfs_unlabelled')
shelf = books / '0016.jpg'
proposed_books = image_model_pytorch(shelf, threshold=0.2, book_limit=100)
found_books = books_from_proposed(proposed_books, display=True)
# test = ['introduction', 'to', 'algorithms']
# cleaned_test = clean_proposals([test])[0]
# print(cleaned_test)
# format_title = format_words_open_library(cleaned_test)
# found_title = get_book_open_library(format_title)
# print(found_title)
# alt_titles = generate_title_possibilities(cleaned_test)
# print(f'alternative titles: {alt_titles}')
# for _, alt_title in enumerate(alt_titles):
#     format_alt_title = format_words_open_library(alt_title)
#     found_alt_title = get_book_open_library(format_alt_title)
#     print(found_alt_title)
