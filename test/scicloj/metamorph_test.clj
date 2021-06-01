(ns scicloj.metamorph-test
  (:require
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.tensor :as dst]
   [tech.v3.tensor :as dtt]

   [tablecloth.api :as tc]
   [libpython-clj2.python :refer [->jvm   py.- py.

                                 python-type   ]]
   [libpython-clj2.require :refer [require-python]]
   [scicloj.sklearn-clj.metamorph :as sklearn-mm]
   [clojure.test :refer [deftest is]]
   [scicloj.metamorph.core :as morph]
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
              ((sklearn-mm/fit-transform :feature-extraction.text :Count-Vectorizer {}))))

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

(deftest svm-pipe

  (let  [XY
         (->
          (tc/dataset [ [-1, -1], [-2, -1], [1, 1], [2, 1]] {:layout :as-rows})
          (tc/add-column :target [1 1 2 2])
          (ds-mod/set-inference-target :target))

         pipeline
         (morph/pipeline
          (sklearn-mm/estimate :preprocessing :standard-scaler {})
          (sklearn-mm/estimate :svm "SVC" {:gamma "auto"}))


         fitted-pipeline
         (pipeline {:metamorph/data XY
                    :metamorph/mode :fit})


         new-data
         (->
          (tc/dataset [ [-0.8 -1]] {:layout :as-rows} )
          (tc/add-column :target [nil])
          (ds-mod/set-inference-target :target))

         result
         (pipeline
          (merge fitted-pipeline
                 {:metamorph/data new-data
                  :metamorph/mode :transform
                  }))
         ]

    (is (= [1.0] (get-in result [:metamorph/data :target])
           )
        )))


(comment
  (require-python '[numpy :as np]
                  '[sklearn.pipeline :refer [make_pipeline]]
                  '[sklearn.preprocessing :refer [StandardScaler]]
                  '[sklearn.svm :refer [SVC]]
                  )
  (def X (np/array [[-1 -1] [-2 -1 ] [1 1] [2 1]] ))
  (def y (np/array [1 1 2 2]))
  (def clf (make_pipeline (StandardScaler) (SVC :gamma "auto")))
  (py. clf fit X y)
  clf
  (->jvm
   (py. clf predict [[-0.8 -1]]))
  )

