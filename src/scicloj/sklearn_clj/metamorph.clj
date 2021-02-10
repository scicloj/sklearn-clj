(comment

  ;;;  need to be done as very first thing in repl, before loading any ns
  (require '[libpython-clj.python :as py])
  (py/initialize! :python-executable "/home/carsten/.conda/envs/scicloj-data-science-handbook/bin/python"
                  :library-path "/home/carsten/.conda/envs/scicloj-data-science-handbook/lib/libpython3.8.so")

  )


(ns scicloj.sklearn_clj.metamorph
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [tablecloth.api :as tc]

            [sciloj.sklearn-clj :as sklearn]
            ))

(defn fit-or-predict [pipeline-ctx module-kw estimator-class-kw kw-args]
  (def pipeline-ctx pipeline-ctx)
  (let [ds (:dataset pipeline-ctx)]
    (case (:mode pipeline-ctx)
      :fit (assoc pipeline-ctx
                  :sciloj.sklearn.stadapip/model
                  (sklearn/fit ds module-kw estimator-class-kw kw-args ))
      :transform (assoc pipeline-ctx
                        :dataset
                        (sklearn/predict
                         (:dataset pipeline-ctx)
                         (:sciloj.sklearn.stadapip/model pipeline-ctx)
                         kw-args
                         )))))


(comment


  (defn pipeline [ctx]
    (-> ctx
        ((fn [ctx]
           (assoc ctx :dataset
                  (ds-mod/set-inference-target (:dataset ctx) :y))
           ))
        (fit-or-predict :linear-model :linear-regression {})))


  (def fitted
    (pipeline
     {:mode :fit
      :dataset
      (tc/dataset {:x1 [1 2 3]
                   :x2  [7 8 9]
                   :y [5 1 13 ]})}))


  (def prediction
    (pipeline
     (merge fitted
            {:mode :transform
             :dataset
             (tc/dataset {:x1 [-6 1 2 1 -5 ]
                          :x2  [1 2 8 9 -2]
                          :y [0 0 0 0]})})))

  )
