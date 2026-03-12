(ns build
  (:refer-clojure :exclude [test])
  (:require [clojure.tools.build.api :as b] ; for b/git-count-revs
            [org.corfield.build :as bb]))

(def lib 'org.scicloj/sklearn-clj)
; alternatively, use MAJOR.MINOR.COMMITS:
;;(def version (format "0.2.%s" (b/git-count-revs nil)))
(def version "0.6")
(def basis (b/create-basis {:project "deps.edn"}))
(def class-dir "target/classes")


(defn test "Run the tests." [opts]
  (-> opts
      (assoc :aliases [:test :runner])
      (bb/run-tests)))

(defn ci "Run the CI pipeline of tests (and build the JAR)." [opts]
  (-> opts
      (assoc :lib lib :version version :aliases [:test :runner])
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

(defn- pom-template []
  [[:description "Plugin to use sklearn models in metamorph.ml"]
   [:url "https://github.com/scicloj/scicloj.ml.xgboost"]
   [:licenses
    [:license
     [:name "Eclipse Public License - v 1.0"]
     [:url "https://www.eclipse.org/legal/epl-1.0/"]]]
   [:scm
    [:url "https://github.com/scicloj/sklearn-clj"]
    [:connection "scm:git:https://github.com/scicloj/sklearn-clj.git"]
    [:developerConnection "scm:git:https://github.com/scicloj/sklearn-clj.git"]
    [:tag (str "v" version)]]])


(defn generate-pom [_]
  (b/write-pom {:class-dir class-dir
                :target "."
                :lib lib
                :version version
                :basis basis
                :pom-data (pom-template)
                :src-dirs ["src"]}))