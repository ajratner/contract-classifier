import sys
import re
from pymongo import MongoClient
from collections import Counter, defaultdict
import numpy as np
from time import time

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from scipy.sparse import hstack

# Porter stemming is *very* slow- cache gives considerable speedup
class PorterStemmerCache:
  def __init__(self):
    self.ps = PorterStemmer()
    self.cache = defaultdict(lambda: None)

  def stem(self, w):
    stemmed = self.cache[w]
    if stemmed:
      return stemmed
    else:
      stemmed = self.ps.stem(w)
      self.cache[w] = stemmed
      return stemmed

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
    print "Loading data"
    t0 = time()
    collection = self.db.contracts
    self.docs = []

    # load + count all the classes
    class_counts = Counter([doc['class'] for doc in collection.find(fields=['class'])])

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
    print "[Finished in %2f seconds]\n" % (time()-t0,)

  # simple cleanup pre-vectorization of e.g. html tags, spaces, etc
  def preprocess(self, stem=False, min_chars=None):
    print "Preprocessing data, stem=%s, min_chars=%s" % (stem, min_chars)
    t0 = time()
    ps = PorterStemmerCache() if stem else None
    for doc in self.docs:
      doc['body'] = re.sub(r'\s\s+|\n', ' ', re.sub(r'<[^>]+>', ' ', doc['body'])).lower()
      if stem:
        t = (lambda w: len(w) > min_chars) if min_chars else (lambda w: True)
        doc['body'] = " ".join([ps.stem(w) for w in re.findall(r'[a-z]+', doc['body']) if t(w)])
        doc['title'] = " ".join([ps.stem(w) for w in re.findall(r'[a-z]+', doc['title']) if t(w)])
    print "Preprocessed: new size = %d MB" % (self.docs_size_mb(),)
    print "[Finished in %2f seconds]\n" % (time()-t0,)

  # final operations / transformation to numpy vec pre-vectorization
  def finalize(self):
    idx = range(len(self.docs))
    np.random.shuffle(idx)
    self.X = []
    self.Y = []
    for i in idx:
      self.X.append(self.docs[i])
      self.Y.append(self.docs[i]['class'])
      self.docs[i] = None
    self.docs = None

# takes as input a list of (name, Vectorizer) pairs,
# where name is the dictionary key for relevant component of x in X
class StructuredVectorizer:
  def __init__(self, vectorizers):
    self.vectorizers = vectorizers

  def fit_transform(self, X_in):
    X_out = None
    for v in self.vectorizers:
      name = v[0]
      vectorizer = v[1]
      X = vectorizer.fit_transform([x[name] for x in X_in])
      if X_out is None:
        X_out = X
      else:
        X_out = hstack([X_out, X])
    return X_out

  def transform(self, X_in):
    X_out = None
    for v in self.vectorizers:
      name = v[0]
      vectorizer = v[1]
      X = vectorizer.transform([x[name] for x in X_in])
      if X_out is None:
        X_out = X
      else:
        X_out = hstack([X_out, X])
    return X_out

# Classification routine
N_FOLDS = 2
def classify(dataset, vectorizer, feature_selector, clf):
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

    n_features = int(0.2*X_train.shape[1])
    print "\tSelecting %s best features..." % n_features
    t1 = time()
    if feature_selector:
      X_train = feature_selector.fit_transform(X_train, Y_train)
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)

    print "\tTraining the classifier..."
    t1 = time()
    clf.fit(X_train, Y_train)
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)

    print "\tTesting..."
    t1 = time()
    X_test = vectorizer.transform([dataset.X[i] for i in test_index])
    if feature_selector:
      X_test = feature_selector.transform(X_test)
    Y_test = np.array([dataset.Y[i] for i in test_index])
    Y_pred = clf.predict(X_test)
    Y_pred_train = clf.predict(X_train)
    score = {"F1": metrics.f1_score(Y_test, Y_pred),
             "Accuracy": metrics.accuracy_score(Y_test, Y_pred),
             "Accuracy (training)": metrics.accuracy_score(Y_train, Y_pred_train),
             "Precision": metrics.precision_score(Y_test, Y_pred),
             "Recall": metrics.recall_score(Y_test, Y_pred)}
    scores.append(score)
    print "\t[Finished in %2f seconds]\n" % (time()-t1,)
    print "[Finished in %2f seconds]\n" % (time()-t0,)
  
  print "SCORES (mean over folds):"
  for k,v in scores[0].iteritems():
    print "%s: %.3f" % (k, np.mean([s[k] for s in scores]))


# run the classifier test(s)
# TODO: basic setup of data loading + classifier execution / testing ------------------> DONE!

# TODO: (put this on github!) ---------------------------------------------------------> DONE!
# TODO: by-class subsampling: load all ids+class, determine classes, subsample --------> DONE!
# TODO: simple preprocessing: stemming, tfidf/thresholding, etc -----------------------> DONE!
# TODO: <run new round of testing here> -----------------------------------------------> DONE!
# TODO: feature formats: all text, titles only, body+titles ---------------------------> DONE!
# TODO: add more than just f1-score ---------------------------------------------------> DONE!

# TODO: question: why is tfidf doing worse?  explore this
# TODO: dimensionality reduction: Chi^2, LSA

# TODO: word2vec vectors...

# TODO: DISCRIMINATIVE classifier algorithms: Logistic regression, NB, SVM

# TODO: <START MAKING POSTER!>

# TODO: structure preprocessing: heuristic title / section extraction

# TODO: (move this to markdown?)

# TODO: [GENERATIVE classifier algorithms: LDA, custom generative model]

# TODO: [Deep learning classifier algorithms: RAE]
if __name__ == '__main__':

  # load data
  dataset = Dataset()
  dataset.load(5000, min_per_class=50)
  dataset.preprocess(stem=True, min_chars=3)
  dataset.finalize()

  # select classification pipeline components 
  #vectorizers = [
  #  ["Hashing", HashingVectorizer(stop_words='english', non_negative=True)],
  #  ["Tfidf", TfidfVectorizer(stop_words='english')],
  #  ["Tfidf", TfidfVectorizer(stop_words='english', min_df=5, max_df=0.5)]
  #]
  vectorizers = [
    #[
    #  "Title only (Hashing)",
    #  StructuredVectorizer([
    #    ["title", HashingVectorizer(stop_words='english', non_negative=True)]
    #  ])],
    #[
    #  "Body only (Hashing)",
    #  StructuredVectorizer([
    #    ["body", HashingVectorizer(stop_words='english', non_negative=True)]
    #  ])],
    [
      "Title + body (Hashing)",
      StructuredVectorizer([
        ["title", TfidfVectorizer(stop_words='english', sublinear_tf=True, use_idf=True)],
        ["body", TfidfVectorizer(stop_words='english', min_df=5, max_df=0.5, sublinear_tf=True, use_idf=True)]
      ])]
  ]
  feature_selectors = [
    ["None", None],
    ["Chi^2", SelectKBest(chi2)],
    ["PCS", PCA(n_components=1000)]
  ]
  classifiers = [
    ["Logistic Regression", LogisticRegression()],
    ["LinearSVC", LinearSVC(loss="l2", penalty="l2", dual=False, tol=1E-3)],
    ["MultinomialNB", MultinomialNB()],
    ["SVC", SVC()]
  ]
  for v in vectorizers:
    for fs in feature_selectors:
      for c in classifiers:
        print "*"*80
        print "Vectorizer: %s, Feature selector: %s, Classifier: %s" % (v[0], fs[0], c[0])
        print "*"*80
        classify(dataset, v[1], fs[1], c[1])
