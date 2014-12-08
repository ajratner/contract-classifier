(ns contracts.onecle-crawl
  (:require [net.cgrand.enlive-html :as html]
            [monger.core :as mg]
            [monger.collection :as mc]
            [clojure.core.async :as async :refer [go <!! >!! chan]])
  (:import java.io.FileNotFoundException))

(def user-agent-0 "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36")

(defn fetch-url
  "Simple synchronous request & parsing of html page w/ Enlive + custom user-agent"  
  ([url]
    (fetch-url url user-agent-0))
  ([url user-agent]
    (with-open [inputstream (-> (java.net.URL. url)
                                .openConnection
                                (doto (.setRequestProperty "User-Agent" user-agent))
                                .getContent)]
      (html/html-resource inputstream))))

(defn rel-to-abs
  [url]
  (if (re-matches #"/.*" url) (str "http://contracts.onecle.com" url) url))

(defn get-onecle-contracts-of-type
  "Scrape all contracts of a certain onecle type index t & save to mongodb db"
  [t]
  (let [index-page (try 
                     (fetch-url (str "http://contracts.onecle.com/type/" t ".shtml"))
                     (catch FileNotFoundException e (do (println e) nil)))]
    (if (not (nil? index-page))
      (doall (remove nil?
        (map
          #(do
            ;(Thread/sleep (* (rand) 1000))
            (try
              (assoc {} :url (-> % :attrs :href rel-to-abs)
                        :title (-> % :content first)
                        :class t
                        :html (-> % :attrs :href rel-to-abs slurp))
              (catch FileNotFoundException e (do (println e) nil))))
          (html/select index-page [:div.index :a]))))
      '())))

;; NOTE: currently on type 30
(defn get-onecle-contracts
  ([start]
    (let [conn (mg/connect)
          db (mg/get-db conn "contracts")
          coll "contracts-raw"]
      (loop [t start]
        (println ">> Loop " t ": waiting...")
        (Thread/sleep (* (+ 1.0 (rand)) 10000))
        (println "Pulling and inserting docs...")
        (let [docs (get-onecle-contracts-of-type t)]
          (if (> (count docs) 0) (mc/insert-batch db coll docs)))
        (println (mc/count db coll) " docs in collection.")
        (if (< t 326) (recur (inc t))))
      (println "Disconnecting...")
      (mg/disconnect conn)))
  ([] (get-onecle-contracts 1)))
