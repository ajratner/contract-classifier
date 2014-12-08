(ns contracts.weka
  (require [clojure.tools.logging :refer [debug error info trace warn]]
           [clojure.string :as s])
  (:import (java.lang.NumberFormatException)
           (java.util ArrayList)
           (weka.core Attribute
                      Instances
                      SparseInstance)
           (weka.classifiers.bayes NaiveBayes)
           (weka.classifiers Evaluation)))


;; Simple machine learning using Weka API

(def BC ["false" "true"])

(defn create-instance-list
  "Creates a new Instances object with Instance size n, with last index for class"
  [inst-name n class-labels]
  (let [atts (ArrayList.) classes (ArrayList.)]
    (doseq [c class-labels] (.add classes c))
    (doseq [att (range n)] (.add atts (Attribute. (str att))))
    (.add atts (Attribute. "class" classes))
    (let [insts (Instances. inst-name atts 0)]
      (.setClassIndex insts n)
      insts)))

(defn make-inst
  [v c]
  (SparseInstance. (double 1) (double-array (conj v c))))

(defn add-instance
  "Adds a dense numeric clojure vector + an index (int) class label to an instances list"
  [v c insts]
  (.add insts (make-inst v c)))

(defn get-training-data
  "Assembles training data using a vectorizer, vectorize fn, a text source fn & a class fn"
  ([data vectorizer vfn get-text get-class v-size class-labels]
    (let [insts (create-instance-list "training-data" v-size class-labels)]
      (doseq [d data]
        (add-instance (vfn (get-text d) vectorizer) (get-class d) insts))
      insts))

  ([data vectorizer vfn get-text get-class v-size]
    (get-training-data data vectorizer vfn get-text get-class v-size BC))

  ([data vectorizer vfn get-text get-class]
    (get-training-data data vectorizer vfn get-text get-class (:vocab-size vectorizer) BC)))

(defn train-classifier
  "Returns a trained classifier for pages based on MySQL data"
  [training-data]
  (let [classifier (NaiveBayes.)]
    (.buildClassifier classifier training-data)
    classifier))

(defn evaluate-classifier
  "Evaluates the classifier using cross-validation on the training data & prints accuracy"
  [trainingData]
  (let [evaluator (Evaluation. trainingData)]
    (.crossValidateModel evaluator "NaiveBayes" trainingData 10 nil (java.util.Random. 1))
    (println (str "\t\tAccuracy: " (.pctCorrect evaluator)))
    evaluator))
