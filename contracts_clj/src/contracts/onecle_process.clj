(ns contracts.onecle-process
  (:require [net.cgrand.enlive-html :as html]
            [monger.core :as mg]
            [monger.collection :as mc]
            [contracts.nlp :as nlp]))

(defn category-counts
  "Get the counts of each type label from the db"
  [db coll]
  (doall (map #(mc/count db coll {:class (+ % 1)}) (range 326))))

(defn extract-body
  "Gets the actual contract from the generic onecle page"
  [html]
  (->> html
       (re-matches #"(?s).*<h4>\s*Sponsored Links\s*</h4>\s*<p.*?</p>(.*)")
       second
       (re-matches #"(?s)(.*)</td>\s*</tr>\s*</table>\s*</div>\s*<div id=\"footer.*")
       second))

(defn html-format?
  "Determines whether the document is html format or old <pre> format"
  [html]
  (if (re-find #"<div|<p(\s|>)" html) true false))

(def MIN_CLASS_SIZE 10)

(defn pipeline
  []
  (let [conn (mg/connect)
        db (mg/get-db conn "contracts")
        coll "contracts-raw"
        coll-new "contracts"
        counts (category-counts db coll)]
    (doseq [doc (mc/find-maps db coll)]

      ;; only take classes with > <MIN_CLASS_SIZE> members
      (if (> (nth counts (- (:class doc) 1)) MIN_CLASS_SIZE)
        (let [body (try (->> doc :html extract-body) 
                        (catch Exception e (do (println e) nil)))]

          ;; insert if body was successfully extracted *and is html-format*
          ;(if (and (not (nil? body)) (html-format? body))
          (if (not (nil? body))
            (mc/insert db coll-new {:title (:title doc)
                                    :url (:url doc)
                                    :class (:class doc)
                                    :body body})))))
    (mg/disconnect conn)))

(defn load-and-strip
  "Loads the contracts and just strips away all html"
  []
  (let [conn (mg/connect) db (mg/get-db conn "contracts") coll "contracts"]
    (map #(merge % {:body (nlp/remove-html (:body %))}) (mc/find-maps db coll))))
    
