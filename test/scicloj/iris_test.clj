(ns scicloj.iris-test 
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.ml.core :as ml]
   [scicloj.ml.metamorph :as mm]
   [scicloj.ml.dataset :as ds]
   
   [scicloj.sklearn-clj.ml])) 



(def iris
  (->
   (ds/dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))


(def pipe-fn
  (ml/pipeline
   (mm/set-inference-target :species)
   (mm/categorical->number [:species])
   {:metamorph/id :model}
   (mm/model {:model-type :sklearn.classification/ridge-classifier
              :predict-proba?  false})))


(def evals
  (ml/evaluate-pipelines [pipe-fn]
                         (ds/split->seq iris)
                         ml/classification-accuracy
                         :loss))

(deftest emtric-is-good
  (is (< 0.5
         (-> evals first first :test-transform :mean))))


