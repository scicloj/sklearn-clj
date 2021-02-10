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
