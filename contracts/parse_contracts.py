import sys
import re
from pymongo import MongoClient

# Handles db connection, loading and pre-processing of the contracts
class Docs:
  def __init__(self, limit=None):
    self.client = MongoClient()
    self.db = self.client.contracts
    self.docs = []
    self.load(limit)

  # return the size of the data
  def docs_size_mb(self):
    return sum([sys.getsizeof(doc['body']) for doc in self.docs]) / 1E6

  # load docs
  def load(self, limit):
    collection = self.db.contracts
    self.docs = [doc for doc in collection.find(limit=limit)]
    print "Loaded %d docs, %d MB" % (len(self.docs), self.docs_size_mb())

def parse_sections(text):
  
  # <pre>-formatted docs
  if re.findall(r'<pre>', text, flags=re.I):
    sections = re.split(r'\n\n+', text)
  else:
    
    # find the most prevalent html container, split on this
    div_count = re.findall(r'<div>|<div [^>]+>', text, flags=re.I)
    p_count = re.findall(r'<p>|<p [^>]+>', text, flags=re.I)
    if div_count > p_count:
      sections = re.findall(r'<div[^>]*>.*?</div>', text, flags=re.I|re.S)
    else:
      sections = re.findall(r'(?:<p>|<p [^>]+>).*?</p>', text, flags=re.I|re.S)

  # conglomerate
  sections_merged = []
  p = []
  for section in sections:
    cleaned = " ".join(re.findall(r'[a-z]+', re.sub(r'<[^>]+>', ' ', section), flags=re.I))
    if len(cleaned) < 100:
      p.append(cleaned)
    else:
      s = " ".join(p)+" "+cleaned if len(p) > 0 else cleaned
      sections_merged.append(s)
      p = []
  return sections_merged
