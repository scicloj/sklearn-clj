(ns scicloj.regression-test

  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.ml.loss :as loss]
            [scicloj.metamorph.core :as mm]
            [tablecloth.api :as tc]
            [scicloj.sklearn-clj :as sklearn]
            [scicloj.sklearn-clj.ml])) 


(deftest test-options-with-number
  (is (= "ElasticNet(alpha=1, l1_ratio=2)"
         (->
          ( sklearn/make-estimator :sklearn.linear-model :elastic-net 
            {:alpha 1
             :l1-ratio 2 })
          str)))
)



(defn validate-regressor-mean [model-type]
  (println model-type)
  (let [  diabetes (toydata/diabetes-ds)
        pipe-fn
        (mm/pipeline
         {:metamorph/id :model}
         (ml/model {:model-type model-type}))
        eval-result
        (ml/evaluate-pipelines [pipe-fn]
                               (tc/split->seq diabetes)
                               loss/rmse
                               :loss)
        ]
    (is (< 50
           (-> eval-result first first :test-transform :mean)))))


(deftest exercise-regressors
  (let [validation-results
        (->>
         (set/difference
          (into #{} (keys @ml/model-definitions*))
          #{:sklearn.regression/multi-task-elastic-net
            :sklearn.regression/multi-task-lasso-cv
            :sklearn.regression/random-forest-regressor
            :sklearn.regression/isotonic-regression
            :sklearn.regression/multi-task-lasso
            :sklearn.regression/multi-task-elastic-net-cv
            :sklearn.regression/extra-trees-regressor
            :sklearn.regression/pls-canonical
            :sklearn.regression/quantile-regressor
            :sklearn.regression/cca})


         (filter #(some? %))
         (filter #(some? (namespace %)))
         (filter #(str/starts-with? (namespace %) "sklearn.regression"))
         (mapv validate-regressor-mean))]
    (is (every?
         true?
         validation-results))))



  
