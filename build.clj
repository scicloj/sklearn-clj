(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]))

(def lib 'scicloj/sklearn-clj)
; alternatively, use MAJOR.MINOR.COMMITS:
;;(def version (format "0.2.%s" (b/git-count-revs nil)))
(def version "0.4.0")

(defn test "Run the tests." [opts]
  (-> opts
      (assoc :aliases [:runner])
      (bb/run-tests)))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version :aliases [:runner])
      (bb/run-tests)
      (bb/clean)
      (bb/jar)))

(defn ci-no-test "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/clean)
      (bb/jar)))
(defn install "Install the JAR locally." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/install)))

(defn deploy "Deploy the JAR to Clojars." [opts]
  (-> opts
      (assoc :lib lib :version version)
      (bb/deploy)))
