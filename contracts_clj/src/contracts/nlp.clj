(ns contracts.nlp
  (require [clojure.tools.logging :refer [debug error info trace warn]]
           [clojure.string :as s])
  (:import (java.lang.NumberFormatException)
           (java.util ArrayList)))


;; Simple first-pass text vectorization functionalities
(defn remove-html
  "Removes all html tags from a string"
  [string]
  (-> string
      (s/replace #"<[^>]+>" " ")
      (s/replace #"&nbsp;" " ")
      (s/replace #"&amp;" "&")
      (s/replace #"\s\s+" " ")))

(defn get-ngrams
  "Parses a string into overlapping subsequences of ordinality n"
  [string n]
  (reduce
    (fn [coll i]
      (into coll (map #(apply str (interpose " " %)) (partition i 1
        (filter #(> (count %) 3) (re-seq #"[A-Za-z]+" (.toLowerCase string)))))))
    '()
    (map inc (range n))))

(defn create-df-token-map
  "Takes in a sequence of documents and creates a dictionary which contains an id and a df"
  [docs]
  (reduce
    (fn [coll toks]
      (reduce
        #(let [x (get %1 %2)]
          (if (nil? x)
            (assoc %1 %2 {:id (count %1) :df 1})
            (merge %1 {%2 {:id (:id x) :df (inc (:df x))}})))
        coll
        (distinct toks)))
    {}
    docs))

(defn filter-df-token-map
  "takes a df token map, filters by min df and returns a normal token map"
  [df-token-map min-df]


(defn create-df-vectorizer
  [strings n]
  (let [vmap (create-df-token-map (map #(get-ngrams % n) strings))]
    {:n n
     :vocab-size (count vmap)
     :vmap vmap}))

(defn create-token-map
  "Takes in a collection of tokens, creates a dictionary vector map"
  [toks]
  (reduce #(if (nil? (get %1 %2)) (assoc %1 %2 (count %1)) %1) {} toks))

(defn create-vectorizer
  "Takes in a collection of strings (docs) and returns an ngram vectorizer"
  [strings n]
  (let [vmap (create-token-map (mapcat #(get-ngrams % n) strings))]
    {:n n
     :vocab-size (count vmap)
     :vmap vmap}))

(defn binary-vectorize
  "Vectorizes a string (doc) using a supplied vectorizer"
  [string v]
  (reduce 
    #(try (assoc %1 (get (:vmap v) %2) 1) 
          (catch Exception e (do
                               (error e "TOKEN MAP ERROR!")
                               (identity %1))))
    (into [] (take (count (:vmap v)) (repeat 0)))
    (get-ngrams string (:n v))))

(defn nil-string?
  "Test for nil or content-empty string"
  [string]
  (or (nil? string) (nil? (re-find #"[A-Za-z0-9]" string))))

(defn get-number-phrases
  "Gets numbers and the words around them with radius r"
  [string r]
  (if (nil-string? string)
    []
    (let [ws (s/split string #"\s+")]
      (loop [coll [] i 0]
        (let [new-coll (if-let [number (re-matches #"($)?\s*(\d+\.?\d*)\s*(%)?" (nth ws i))]
                       (conj coll 
                         {:num number
                          :val (Float. (nth number 2)) 
                          :str (apply str (interpose " " (map #(nth ws %) 
                               (range (max 0 (- i r)) (min (count ws) (+ i r 1))))))})
                       coll)]
          (if (< (+ i 1) (count ws))
            (recur new-coll (inc i))
            new-coll))))))
