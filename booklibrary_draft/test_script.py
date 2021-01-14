import click
from pathlib import Path
from image_model import image_model_pytorch
from text_model import books_from_proposed

# run as command line program
@click.command()
@click.argument('filename', type=click.Path(exists=True))
def shelf_books(filename):
    shelf = click.format_filename(filename)
    click.echo(f'Analyzing bookshelf with path: {shelf}')
    proposed_books = image_model_pytorch(shelf, threshold=0.2, book_limit=100)
    found_books = books_from_proposed(proposed_books, display=False, verbose=False)
    click.echo(found_books)


if __name__ == '__main__':
    shelf_books()
