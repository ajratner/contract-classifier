(defproject contracts "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.5.1"]
                 [org.clojure/core.async "0.1.267.0-0d7780-alpha"]
                 [enlive "1.1.5"]
                 [nz.ac.waikato.cms.weka/weka-dev "3.7.11"]
                 [org.clojure/tools.logging "0.3.0"]
                 [com.novemberain/monger "2.0.0"]]
  :main ^:skip-aot contracts.core
  :target-path "target/%s"
  :profiles {:uberjar {:aot :all}})
