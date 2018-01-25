# distutils: language=c++
# distutils: sources=ngram_language_model_cpp.cpp
# distutils: extra_compile_args=["-std=c++11"]

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef extern from "ngram_language_model_cpp.h" namespace "language":
  cdef cppclass NgramLanguageModelCPP:
    NgramLanguageModelCPP(int, int) except +
    void add_sample(const vector[int]& sample)
    vector[vector[int]] get_unique_ngrams(int n)
    double js_with(const NgramLanguageModelCPP& other, int ngram_length)
    double log_likelihood(const vector[int]& ngram)
    long int get_memory()

cdef class NgramLanguageModel:
  cdef NgramLanguageModelCPP* _nglm
  cdef int _n
  def __cinit__(self, samples, int n, int m):
    '''
    Constructor that takes a corpus of samples and creates a tree like ngram language model from the corpus
    @param samples list of lists (or matrix) of integers from which ngrams are extracted and added to the model
    @param n length of ngrams to be saved (i.e. 1,2 and 3 grams if n=3)
    @param m maximum integer that will be in sample. Having a sample in samples larger than m will lead to undefined behavior
    '''
    if n <= 0:
      raise ValueError('n must be >= 1')
    if m <= 1:
      raise ValueError('m must be >= 2')
    self._n = n
    self._nglm = new NgramLanguageModelCPP(n, m)
    cdef vector[int] v
    for sample in samples:
      v = np.ascontiguousarray(sample, dtype=DTYPE)
      self._nglm.add_sample(v)
      

  def __dealloc__(self):
    del self._nglm

  def unique_ngrams(self, int n):
    '''
    Return list of all ngrams of a specific length
    @param The ngram lengths that should be retuned
    '''
    if n <= 0:
      raise ValueError('n must be >= 1')
    if n > self._n:
      raise ValueError('no ngrams of this size saved in model')
      
    cdef vector[vector[int]] v = self._nglm.get_unique_ngrams(n)
    cdef np.ndarray[DTYPE_t, ndim=2] answer = np.zeros((v.size(),n), dtype=DTYPE)
    cdef int i, j
    for i in range(v.size()):
      for j in range(n):
        answer[i,j] = v[i][j]
    return answer

  def log_likelihood(self, ngram):
    '''
    Get the log likelihood of a specific ngram being in this ngram model
    @param ngram list or array of ints
    @note putting in ints that are larger than the maximum size defined in constructor leads to undefined behavior
    '''
    cdef vector[int] v = np.ascontiguousarray(ngram, dtype=DTYPE)
    return self._nglm.log_likelihood(v)

  def js_with(self, NgramLanguageModel other, int ngram_length):
    '''
    Return the jensen shannon distance between two ngram models for ngrams of a secific length
    @param other NgramLanguageModel to compare to this NgramLanguageModel
    @param ngram_length length of the ngrams where the JS distance is calculated
    '''
    if ngram_length <= 0:
      raise ValueError('ngram_length must be >= 1')
    if ngram_length > self._n:
      raise ValueError('no ngrams of this size saved in this model')
    if ngram_length > other._n:
      raise ValueError('no ngrams of this size saved in other model')
    return self._nglm.js_with((other._nglm)[0], ngram_length)
    
  def get_memory(self):
    return self._nglm.get_memory()



