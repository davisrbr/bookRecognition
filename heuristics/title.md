# Title Text Heuristics
Potential heuristics for grabbing title/author/publisher text from a book with an OCR model.

### Format 
Each heuristic has a format:
 452. *Title of heuristic* (potentially relevant model): minimal real example -> proposed title outputs *Implemented/not implemented*

### Heuristics

1. *Cut first and final words (or first 2, last 2), which tend to be authors/publishers/misses* (keras-ocr): ['haynes', 'probability', 'theory', 'canning'] -> ['probability', 'theory'] *Implemented*

2. *Rotate book and take text from 'best' rotation (using some metric)* (keras-ocr): ['friedman', 'basic', 'books'] -> ['special', 'relativity', 'and', 'classical', 'field', 'theory'] *Implemented*

3. *Rotate book and combine text from rotations* (keras-ocr) *Not implemented*

4. *Sometimes model grabs multiple books: separate them by spatial position* (keras-ocr): ['chomsky', 'syntactic', 'structures', 'discrete', 'mathematics', 'in', 'computer', 'science'] -> ['chomsky', 'syntactic', 'structures'], ['discrete', 'mathematics', 'in', 'computer', 'science']  *Not implemented*
