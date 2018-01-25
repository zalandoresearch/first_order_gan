from language_helpers import NgramLanguageModel
import ngram_language_model
import numpy as np

def corpus_1():
  c = []
  c.append('cat in hat')
  c.append('strange pe')
  return c

def corpus_2():
  c = []
  c.append('lead a hor')
  c.append('love to ea')
  return c

def c2m(c):
  answer = np.zeros((len(c), len(c[0])) ,dtype=np.int32)
  for i in answer.shape[0]:
    for j in answer.shape[1]:
      answer[i,j] = ord(c[i][j]) - ord(' ')
  return answer

def decode(m):
  '''
  take a matrix with a bunch of n-grams and transform it to a set of strings
  '''
  answer = []
  for s in m:
    answer.append('')
    for c in s:
      answer[-1] += chr(c + ord(' '))
  return set(answer)

def convert(s):
  if type(s[0]) == type(''):
    answer = np.zeros((len(s), len(s[0])) ,dtype=np.int32)
    for i,l in enumerate(s):
      for j,c in enumerate(l):
        answer[i,j] = ord(c) - ord(' ')
    return answer
  else:
    answer = np.zeros(len(s) ,dtype=np.int32)
    for i,c in enumerate(s):
      answer[i] = ord(c) - ord(' ')
    return answer

def test_counts():
  corpus = convert(corpus_1())
  fast = ngram_language_model.NgramLanguageModel(corpus, 4, ord('z') - ord(' '))
  for i in range(4):
    slow = NgramLanguageModel(i+1, corpus_1())
    assert sorted(slow.unique_ngrams()) == sorted(decode(fast.unique_ngrams(i+1)))
  print('passed test_counts()')

def test_log_likelihood():
  corpus = convert(corpus_1())
  fast = ngram_language_model.NgramLanguageModel(corpus, 4, ord('z') - ord(' '))
  for i in range(4):
    slow = NgramLanguageModel(i+1, corpus_1())
    unique = slow.unique_ngrams()
    for ngram in unique:
      assert np.abs(slow.log_likelihood(ngram) - fast.log_likelihood(convert(ngram))) < 10e-8

  for i in range(4):
    slow = NgramLanguageModel(i+1, corpus_2())
    unique = slow.unique_ngrams()
    slow = NgramLanguageModel(i+1, corpus_1())
    for ngram in unique:
      sll = slow.log_likelihood(ngram)
      fll = fast.log_likelihood(convert(ngram))
      assert (sll == -np.inf and fll == -np.inf) or np.abs(sll - fll) < 10e-8

  print('passed test_log_likelihood()')

def test_js_with():
  fast1 = ngram_language_model.NgramLanguageModel(convert(corpus_1()), 4, ord('z') - ord(' '))
  fast2 = ngram_language_model.NgramLanguageModel(convert(corpus_2()), 4, ord('z') - ord(' '))
  for i in range(4):
    slow1 = NgramLanguageModel(i+1, corpus_1())
    slow2 = NgramLanguageModel(i+1, corpus_2())
    assert(np.abs(slow1.js_with(slow1) - fast1.js_with(fast1, i+1)) < 10e-8)
    assert(np.abs(slow1.js_with(slow2) - fast1.js_with(fast2, i+1)) < 10e-8)
    assert(np.abs(slow2.js_with(slow1) - fast2.js_with(fast1, i+1)) < 10e-8)
    assert(np.abs(slow2.js_with(slow2) - fast2.js_with(fast2, i+1)) < 10e-8)
  print('passed test_js_with()')
     
def main():
  test_counts()
  test_log_likelihood()
  test_js_with()

if __name__=='__main__':
  main()
