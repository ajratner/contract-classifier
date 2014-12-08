(ns contracts.core
  (:require [contracts.nlp :as nlp]
            [contracts.weka :as weka]
            [contracts.onecle-process :as process])
  (:gen-class))

(defn load-data [] (doall (process/load-and-strip)))

(defn -main
  "I don't do a whole lot ... yet."
  [& args]
  (println "Hello, World!"))
