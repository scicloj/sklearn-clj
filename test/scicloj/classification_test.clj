(ns scicloj.classification-test
  (:require
   [clojure.set :as set]
   [clojure.string :as str]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.sklearn-clj.ml]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as ds-mm]))

(def iris
  (->
   (ds/->dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))

(defn- train-eval [model-type]
  (let [pipe-fn
        (mm/pipeline
         (ds-mm/set-inference-target :species)
         (ds-mm/categorical->number [:species] [] :int64)
         {:metamorph/id :model}
         (ml/model {:model-type model-type}))


        evals
        (ml/evaluate-pipelines [pipe-fn]
                               (tc/split->seq iris)
                               loss/classification-accuracy
                               :loss)]
    (-> evals first first :test-transform :mean)))

(deftest exercise-classifiers
  (->>
   (set/difference
    (into #{} (keys @ml/model-definitions*))
    #{:sklearn.classification/self-training-classifier})


   (filter #(some? %))
   (filter #(some? (namespace %)))
   (filter #(str/starts-with? (namespace %) "sklearn.classification"))
   (run! #(do
            (print :model-type % " ...")
            (flush)
            (let [accuracy (train-eval %)]
              (println " : " accuracy)
              (if (contains? #{:sklearn.classification/bernoulli-nb
                               :sklearn.classification/dummy-classifier} %)
                (println "Skipping assert for" %)
                (is (< 0.5 accuracy))))))))

