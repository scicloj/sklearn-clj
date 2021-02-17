(ns scicloj.sklearn-clj.metamorph
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.sklearn-clj :as sklearn]
            ))

(defn estimate [module-kw estimator-class-kw kw-args]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
      (case mode
        :fit (assoc ctx
                    id
                    (sklearn/fit data module-kw estimator-class-kw kw-args ))
        :transform (assoc ctx
                          :metamorph/data
                          (sklearn/predict data
                                           (get ctx id)
                                           kw-args)))))


(defn transform [module-kw estimator-class-kw kw-args]
  (fn [{:metamorph/keys [data id] :as ctx}]
    (let [{:keys [ds estimator]} (sklearn/fit-transform data module-kw estimator-class-kw kw-args )]
      (assoc ctx
             :metamorph/data ds
             id estimator
             ))))
