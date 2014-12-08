import sys
import re
from pymongo import MongoClient
from collections import Counter, defaultdict
import numpy as np
from time import time

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC

# UTILS


# Handles db connection, loading and pre-processing of the contracts
class Dataset:
  def __init__(self):
    self.client = MongoClient()
    self.db = self.client.contracts
    self.docs = []
    self.X = None
    self.Y = None

  # return the size of the data
  def docs_size_mb(self):
    return sum([sys.getsizeof(doc['body']) for doc in self.docs]) / 1E6

  # load + subsample, stratified by class
  def load(self, n, min_per_class=None, n_classes=None):
    collection = self.db.contracts
    self.docs = []

    # load + count all the classes
    class_counters = Counter([doc['class'] for doc in collection.find(fields=['class'])])

    # calculate how many to take from each class, opt. threshold by min_per_class or n_classes
    if n_classes is not None:
      classes, counts = zip(*class_counts.most_common(n_classes))
    else:
      t = (lambda x: x > min_per_class) if min_per_class else (lambda x: True)
      classes, counts = zip(*[c for c in class_counts.iteritems() if t(c[1])])
    take_counts = [0 for c in classes]
    total_taken = 0
    n_classes = len(classes)
    for i in range(sum(counts)):
      j = i%n_classes
      if total_taken == n:
        break
      if take_counts[j] < counts[j]:
        take_counts[j] += 1
        total_taken += 1
    print "Taking %d classes..." % n_classes

    # load in docs by class to keep memory footprint low + subsample randomly
    for i,c in enumerate(classes):
      take = np.random.permutation(counts[i])[:take_counts[i]]
      self.docs += [doc for j,doc in enumerate(collection.find({"class": c})) if j in take]
    print "Loaded %d docs, %d MB" % (len(self.docs), self.docs_size_mb())

  # simple cleanup pre-vectorization of e.g. html tags, spaces, etc
  # TODO: add stemming, other simple?
  def simple_preprocess(self):
    for doc in self.docs:
      doc['body'] = re.sub(r'\s\s+|\n', ' ', re.sub(r'<[^>]+>', ' ', doc['body']))
    print "Preprocessed: new size = %d MB" % (self.docs_size_mb(),)

  # pre-process into titles, etc.
  # TODO: implement this!

  # final operations / transformation to numpy vec pre-vectorization
  def finalize(self):
    idx = range(len(self.docs))
    np.random.shuffle(idx)
    self.X = []
    self.Y = []
    for i in idx:
      self.X.append(self.docs[i]['body'])
      self.Y.append(self.docs[i]['class'])
      self.docs[i] = None
    self.docs = None

# Classification routine
def classify(dataset, vectorizer, clf):
  print "Performing K-fold cross validation with %s folds:" % N_FOLDS
  skf = StratifiedKFold(dataset.Y, n_folds=N_FOLDS)
  fold_n = 0
  scores = []
  for train_index, test_index in skf:
    t0 = time()
    fold_n += 1
    print "\nBeginning fold %s:" % fold_n

    print "\tVectorizing training set..."
    t1 = time()
    X_train = vectorizer.fit_transform([dataset.X[i] for i in train_index])
    Y_train = np.array([dataset.Y[i] for i in train_index])
    print "\tn_samples: %d, n_features: %d" % X_train.shape
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)

    print "\tTraining the classifier..."
    t1 = time()
    clf.fit(X_train, Y_train)
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)

    print "\tTesting..."
    t1 = time()
    X_test = vectorizer.transform([dataset.X[i] for i in test_index])
    Y_test = np.array([dataset.Y[i] for i in test_index])
    Y_pred = clf.predict(X_test)
    score = metrics.f1_score(Y_test, Y_pred)
    scores.append(score)
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)

    print "[Finished in %2f seconds]\n" % (time()-t0,)

  print "Final (mean) f1-score: %.3f" % (np.mean(scores),)


# run the classifier test(s)
# TODO: basic setup of data loading + classifier execution / testing ------------------> DONE!

# TODO: (put this on github!)
# TODO: by-class subsampling: load all ids+class, determine classes, subsample --------> DONE!
# TODO: simple preprocessing: stemming, tfidf/thresholding, etc
# TODO: structure preprocessing: heuristic title / section extraction
# TODO: feature formats: all text, titles only, body+titles, other...?

# TODO: question: why is tfidf doing worse?  explore this
# TODO: dimensionality reduction: Chi^2, LSA

# TODO: word2vec vectors...

# TODO: DISCRIMINATIVE classifier algorithms: Logistic regression, NB, SVM

# TODO: <START MAKING POSTER!>

# TODO: [GENERATIVE classifier algorithms: LDA, custom generative model]

# TODO: [Deep learning classifier algorithms: RAE]
if __name__ == '__main__':

  # Globals
  N_DOCS = 5000
  TOP_PERCENT = 0.8
  MIN_NUMBER_PER_CATEGORY = 20
  N_FOLDS = 5
  
  # load data
  dataset = Dataset(N_DOCS)
  dataset.simple_preprocess()
  dataset.top_classes_pm(TOP_PERCENT, MIN_NUMBER_PER_CATEGORY)
  dataset.finalize()

  # select classification pipeline components 
  vectorizers = [
    ["Hashing", HashingVectorizer(stop_words='english', non_negative=True)],
    ["Tfidf", TfidfVectorizer()],
    ["Tfidf", TfidfVectorizer(stop_words='english')],
    ["Tfidf", TfidfVectorizer(stop_words='english', min_df=5, max_df=0.5)]
  ]
  classifiers = [
    ["LinearSVC", LinearSVC(loss="l2", penalty="l2", dual=False, tol=1E-3)]
  ]
  for v in vectorizers:
    for c in classifiers:
      print "*"*80
      print "Vectorizer: %s, Classifier: %s" % (v[0], c[0])
      print "*"*80
      classify(dataset, v[1], c[1])
