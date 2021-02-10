(comment

  ;;;  need to be done as very first thing in repl, before loading any ns
  (require '[libpython-clj.python :as py])
  (py/initialize! :python-executable "/home/carsten/.conda/envs/scicloj-data-science-handbook/bin/python"
                  :library-path "/home/carsten/.conda/envs/scicloj-data-science-handbook/lib/libpython3.8.so")

  )


(ns scicloj.sklearn-clj.metamorph
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.sklearn-clj :as sklearn]
            ))

(defn estimate [ctx module-kw estimator-class-kw kw-args]
  (let [ds (:metamorph/dataset ctx)
        id (:metamorph/id ctx)]
    (case (:metamorph/mode ctx)
      :fit (assoc ctx
                  id
                  (sklearn/fit ds module-kw estimator-class-kw kw-args ))
      :transform (assoc ctx
                        :metamorph/dataset
                        (sklearn/predict ds
                         (get ctx id)
                         kw-args)))))
