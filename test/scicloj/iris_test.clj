(ns scicloj.iris-test 
  (:require
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as ds-mm]
   [tablecloth.api :as tc]
   [scicloj.sklearn-clj.ml])) 



(def iris
  (->
   (ds/->dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))


(def pipe-fn
  (mm/pipeline
   (ds-mm/set-inference-target :species)
   (ds-mm/categorical->number [:species])
   {:metamorph/id :model}
   (ml/model {:model-type :sklearn.classification/ridge-classifier
              :predict-proba?  false})))


(def evals
  (ml/evaluate-pipelines [pipe-fn]
                         (tc/split->seq iris)
                         loss/classification-accuracy
                         :loss))

(deftest emtric-is-good
  (is (< 0.5
         (-> evals first first :test-transform :mean))))


