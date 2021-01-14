# bookRecognition
A project with the goal of converting an image of a bookshelf to a list of books.

### Goal
We should be able to easily share our bookshelfs with others. I have gotten a lot out of [Patrick Collison's bookshelf](https://patrickcollison.com/bookshelf) [and](http://gordonbrander.com/lib/) [other](http://nickcammarata.com/bookshelf) [similar](https://aaronzlewis.com/starterpack/) [bookshelves](https://tomcritchlow.com/wiki/books/bookshelves/). I want to be able to see and be inspired by the books in others' bookshelves! 


For now, this project is composed of 

1. [An initial attempt with traditional computer vision.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/Traditional_Image_Processing.ipynb)
2. [Experimenting with some optical character recognition systems.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/OCR_simple.ipynb)
3. [Book detection with PyTorch.](https://github.com/davisrbr/bookRecognition/blob/master/notebooks/PyTorch_Detection.ipynb)
4. [A command line interface with the full pipeline](https://github.com/davisrbr/bookRecognition/blob/master/booklibrary_draft). An example script call is `python test_script.py path/to/bookshelf.png`. For now, the pipeline is slow (it, among other things, uses Open Library's API to identify books in a very slow way) and not all that accurate. 
