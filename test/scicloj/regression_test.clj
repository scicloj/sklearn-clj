(ns scicloj.regression-test

  (:require [clojure.set :as set]
            [clojure.string :as str]
            [clojure.test :refer [deftest is]]
            [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.ml.core :as ml]
            [scicloj.ml.dataset :as ds]
            [scicloj.ml.metamorph :as mm]
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
        (ml/pipeline
         {:metamorph/id :model}
         (mm/model {:model-type model-type}))
        eval-result
        (ml/evaluate-pipelines [pipe-fn]
                               (ds/split->seq diabetes)
                               ml/rmse
                               :loss)
        ]
    (is (< 50
           (-> eval-result first first :test-transform :mean)))))


(deftest exercise-regressors 
  (let [ validation-results
        (->>
         (set/difference 
          (into #{} (keys @scicloj.ml.core/model-definitions*))
          #{:sklearn.regression/multi-task-elastic-net
            :sklearn.regression/multi-task-lasso-cv
            :sklearn.regression/random-forest-regressor
            :sklearn.regression/isotonic-regression
            :sklearn.regression/multi-task-lasso
            :sklearn.regression/multi-task-elastic-net-cv
            })
         
         
         (filter #(some? %))
         (filter #(some? (namespace %)))
         (filter #(str/starts-with? (namespace %) "sklearn.regression"))
         (mapv validate-regressor-mean)
         )]
    (is (every? 
         true?
         validation-results
               )))
  )



  
