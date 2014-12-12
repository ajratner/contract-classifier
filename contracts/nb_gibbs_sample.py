import numpy as np
import sys
import re
from time import time
from pymongo import MongoClient
from collections import defaultdict

from classify_contracts import Dataset

from sklearn.feature_extraction.text import CountVectorizer

def sample_log_dirichlet(p):
  return np.log(np.random.dirichlet(p, 1)[0])

def sample_log_multinomial(p):
  return np.random.multinomial(1, np.exp(p)).argmax()

def sample_multinomial(p):
  return np.random.multinomial(1, p).argmax()

def word_seq(vec):
  s = []
  for i,count in enumerate(vec):
    for j in xrange(count):
      s.append(i)
  return s

def log_sum_exp(log_vals):
  out = np.array([-np.inf])
  for x in log_vals[np.nonzero(log_vals != -np.inf)]:
    np.logaddexp(out, x, out)
  return out[0]

def remap_idx(X):
  l = len(set(X))
  idx = defaultdict(lambda: None)
  X_out = np.zeros(len(X), dtype=int)
  for i,x in enumerate(X):
    if idx[x] is None:
      idx[x] = len(idx) - 1
    X_out[i] = idx[x]
  return X_out

class GibbsSampler:
  """
  - We sample a prior distribution over classes p(c), theta_c ~ Dirichlet(gamma)
  - We sample a prior distribution over words p(w|c), theta_w_c ~ Dirichlet(gamma_c)
  - For each document d_i:
    * We sample a class c_i, c_i ~ Multinomial(theta_c)
    * For each word w_1,...,w_R_i in d_i:
      + We sample a word w_j ~ Multinomial(theta_w_c)

  Simplifying assumptions:
  - We assume symmetric uniform Dirichlet priors, i.e. gamma, gamma_c = vec(1)

  We keep track of the following variables:
  - doc_class[i] = the class of d_i
  - class_doc_counts[j] = the count of docs of class c_j
  - class_word_counts[w,j] = the count of words
  """
  
  def __init__(self, X, y, holdout=0.25):
    print "Initializing..."
    t0 = time()
    y = remap_idx(y)
    
    # split into train and test
    self.n_docs, self.vocab_size = X.shape
    self.n_classes = len(set(y))
    idx = np.random.permutation(self.n_docs)
    self.X_train = X[idx[int(holdout*self.n_docs):]]
    self.y_train = y[idx[int(holdout*self.n_docs):]]
    self.X = X[idx[:int(holdout*self.n_docs)]]
    self.y = y[idx[:int(holdout*self.n_docs)]]

    # initialize the counts
    self.doc_class = np.zeros(self.n_docs, dtype=int)
    self.class_doc_counts = np.zeros(self.n_classes)
    self.class_word_counts = np.zeros((self.n_classes, self.vocab_size))
    self.log_thetas = np.empty((self.n_classes, self.vocab_size))
    self.class_assignments = np.zeros((self.n_docs, self.n_classes), dtype=int)

    # training data
    for i,x in enumerate(self.X_train):
      c = self.y_train[i]
      self.doc_class[i+int(holdout*self.n_docs)] = c
      self.class_doc_counts[c] += 1
      self.class_word_counts[c] += x

    # test data - use the training data counts as theta_c
    for i,x in enumerate(self.X):
      c = sample_multinomial(self.class_doc_counts / sum(self.class_doc_counts))
      self.doc_class[i] = c
      self.class_doc_counts[c] += 1
      self.class_word_counts[c] += x

    # sample the theta_w_c based on all data
    for c in xrange(self.n_classes):
      self.log_thetas[c] = sample_log_dirichlet(self.class_word_counts[c])
    print "[Finished in %2f seconds]\n" % (time()-t0,)

  # get the classes of the documents
  def get_classes(self):
    return np.argmax(self.class_assignments, 1)

  def get_accuracy(self):
    n_test_docs = self.X.shape[0]
    y_pred = self.get_classes()[:n_test_docs]
    score = 0.0
    for i,y in enumerate(y_pred):
      score += 1.0  if y == self.y[i] else 0.0
    return score / n_test_docs

  def run_iteration(self):
    for i,x in enumerate(self.X):

      # subtract counts related to this document
      c = self.doc_class[i]
      self.class_doc_counts[c] -= 1
      self.class_word_counts[c] -= x

      # sample a new class for this document
      log_cond_p_c = np.repeat(-np.inf, self.n_classes)
      for ci in xrange(self.n_classes):
        log_p_c = np.log(self.class_doc_counts[ci] / self.n_docs)
        wp = self.log_thetas[ci] * x
        log_p_c_w = (wp[~np.isnan(wp)]).sum()
        log_cond_p_c[ci] = log_p_c + log_p_c_w
      log_cond_p_c -= log_sum_exp(log_cond_p_c)

      # NOTE: why is sum(exp(log_cond_p_c)) > 1.0 (e.g. = 1.0000000001) sometimes?
      sum_p = sum([np.exp(lpc) for lpc in log_cond_p_c])
      if sum_p > 1.0:
        log_cond_p_c -= np.log(sum_p)
      c_new = sample_log_multinomial(log_cond_p_c)

      # >>> TESTING
      #if c != c_new:
      #  print "doc %s: %s --> %s" % (i, c, c_new)

      # add counts related to this document to the new category
      self.doc_class[i] = c_new
      self.class_doc_counts[c_new] += 1
      self.class_word_counts[c_new] += x

    # resample theta_w_c
    for c in xrange(self.n_classes):
      self.log_thetas[c] = sample_log_dirichlet(self.class_word_counts[c])

  def run(self, iters=20, burn_in=0, lag=0):
    maxiter = iters*(1+lag) + burn_in
    lag_counter = lag
    for it in xrange(maxiter):
      print "Iteration %s..." % it
      t0 = time()
      self.run_iteration()

      # add to class assignment counts if past burn-in & not in lag
      if burn_in > 0:
        burn_in -= 1
      elif lag_counter > 0:
        lag_counter -= 1
      else:
        lag_counter = lag
        for i,c in enumerate(self.doc_class):
          self.class_assignments[i,c] += 1
        print "Iteration %s: Accuracy = %.3f" % (it, self.get_accuracy())
      print "[Finished in %2f seconds]\n" % (time()-t0,)

if __name__ == "__main__":

  # Load data
  print "Loading data..."
  dataset = Dataset()
  dataset.load(1000, n_classes=2)
  dataset.preprocess(stem=True, min_chars=3)
  dataset.finalize()

  # vectorize
  print "Vectorizing..."
  vectorizer = CountVectorizer(stop_words='english', min_df=3, max_df=0.8)
  X = np.array(vectorizer.fit_transform([doc['body'] for doc in dataset.X]).todense())
  y = np.array(dataset.Y)
  print "n_docs = %s, vocab_size = %s\n" % X.shape

  # run gibbs sampling
  print "Running gibbs sampling..."
  sampler = GibbsSampler(X, y)
  sampler.run(iters=25, burn_in=25, lag=2)
