(ns scicloj.metamorph-test

  (:require [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset :as ds]
          [scicloj.sklearn-clj.metamorph :as sklearn-mm]
          [clojure.test :refer [deftest is]]
          )

  )
(defn pipeline [ctx]
    (-> ctx
        ((fn [ctx]
           (assoc ctx :metamorph/dataset
                  (ds-mod/set-inference-target (:metamorph/dataset ctx) :y))
           ))
        (sklearn-mm/estimate :linear-model :linear-regression {})))

(deftest test-estimate
  (let [fitted
        (pipeline

         {:metamorph/id "1"
          :metamorph/mode :fit
          :metamorph/dataset
          (ds/->dataset {:x1 [1 1 2 2]
                       :x2  [1 2 2 3]
                       :y [6 8 9 11 ]})})


        prediction
        (pipeline
         (merge fitted
                {:metamorph/mode :transform
                 :metamorph/dataset
                 (ds/->dataset {:x1 [3]
                              :x2  [5 ]
                              :y [0]})}))]

    (is (= 16
           (Math/round
            (first (seq (get-in prediction [:metamorph/dataset :y] ))))))
    )

  )
