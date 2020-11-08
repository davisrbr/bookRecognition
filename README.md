# bookRecognition
A project with the goal of converting an image of a bookshelf to a list of books.

### Goal
We should be able to easily share our bookshelfs with others. I have gotten a lot out of [Patrick Collison's bookshelf](https://patrickcollison.com/bookshelf) [and](http://gordonbrander.com/lib/) [other](http://nickcammarata.com/bookshelf) [similar](https://aaronzlewis.com/starterpack/) [bookshelves](https://tomcritchlow.com/wiki/books/bookshelves/). I want to be able to see and be inspired by the books in others' bookshelves! 


For now, this project is composed of 

1. [An initial attempt with traditional computer vision.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/Traditional_Image_Processing.ipynb)
2. [Experimenting with some optical character recognition systems.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/OCR_simple.ipynb)
3. [Book detection with PyTorch.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/PyTorch_Detection.ipynb)

and [a command line script implementing the basic pipeline](https://github.com/davisrbr/bookRecognition/tree/master/booklibrary_draft):
run with e.g. `python test_script.py bookRecognition/bookshelfs_unlabelled/0024.jpg`
