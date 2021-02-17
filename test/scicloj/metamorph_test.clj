(ns scicloj.metamorph-test

  (:require [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
            [scicloj.sklearn-clj.metamorph :as sklearn-mm]
            [clojure.test :refer [deftest is]]
            )

  )


(deftest test-estimate
  (let [pipeline
        (fn [ctx]
          (-> ctx
              ((fn [ctx]
                 (assoc ctx :metamorph/data
                        (ds-mod/set-inference-target (:metamorph/data ctx) :y))
                 ))
              ((sklearn-mm/estimate :linear-model :linear-regression {}))))

        fitted
        (pipeline

         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/data
          (ds/->dataset {:x1 [1 1 2 2]
                         :x2  [1 2 2 3]
                         :y [6 8 9 11 ]})})


        prediction
        (pipeline
         (merge fitted
                {:metamorph/mode :transform
                 :metamorph/data
                 (ds/->dataset {:x1 [3]
                                :x2  [5 ]
                                :y [0]})}))]

    (is (= 16
           (Math/round
            (first (seq (get-in prediction [:metamorph/data :y] ))))))))


(deftest test-transform
  (let [pipeline
        (fn [ctx]
          (-> ctx
              ((sklearn-mm/transform :feature-extraction.text :Count-Vectorizer {}))))

        fitted
        (pipeline

         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/data
          (ds/->dataset {:text ["hellow world"]})})

        ]
    (is (= [1]
           (get-in fitted [:metamorph/data 1] )))

    )

  )
