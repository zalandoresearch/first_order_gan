
This project defines a python and C++ API to the class NgramLanguageModel which saves ngrams efficiently and can quickly calculate statistics (like jensen shannon distance between two models). It's written in C++ for speed and has a convenient python wrapper for convenience.

This project was based on the NgramLanguageModel class found in the improved
wasserstein gan training code which can be found here
https://github.com/igul222/improved_wgan_training

Setup:
------

To compile, simply run:

python setup.py build_ext --inplace

set the python path to find the .so file, and import normally

The Language Model
------------------
This code converts texts into an Ngram model. That means a text is analyzed and
the ngrams present are counted and saved. With this model all present ngrams
can be accessed, or the distribution of ngrams between different models can be
analyzed (for example the jensen shannon distance.

If we analyze the following text: 'the cat ate the bat' we can build 1 and 2
gram models

*1-gram model:*

| 1-gram  | frequency |
|---------|-----------|
| t       | 5         |
| h       | 2         |
| e       | 3         |
| [space] | 4         |
| c       | 1         |
| a       | 3         |
| b       | 1         |

*2-gram model:*

| 2-gram   | frequency |
|----------|-----------|
| th       | 2         |
| he       | 2         |
| e[space] | 3         |
| [space]c | 1         |
| ca       | 1         |
| at       | 3         |
| t[space] | 1         |
| [space]a | 1         |
| te       | 1         |
| [space]t | 1         |
| [space]b | 1         |
| ba       | 1         |

In order to efficiently store the ngrams, it's helpful to save them in a tree
format. for example, if to store the 3-grams 'cat', 'car' and 'bat', they can
be saved like this

        |
       / \
      /   \
     /     \
     |     |
     b     c
     |     |
     a     a
     |    / \
     t    t  r

Then searching to see if a some other 3gram is in the model takes has a complexity
of O(3) and the model is very memory efficient. This is achieved with some efficient
C++ pointer magic and is much faster than the original python implementation

Example:
--------

See test_language_model.py for a few examples.


