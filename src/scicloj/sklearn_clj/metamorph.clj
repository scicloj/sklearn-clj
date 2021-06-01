(ns scicloj.sklearn-clj.metamorph
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.sklearn-clj :as sklearn]

            [libpython-clj2.python :refer [dir ->jvm  py.- py. python-type]]
            ))

(defn estimate [module-kw estimator-class-kw kw-args]
  (fn [{:metamorph/keys [id data mode] :as ctx}]
      (case mode
        :fit (assoc ctx
                    id
                    (sklearn/fit data module-kw estimator-class-kw kw-args ))
        :transform (assoc ctx
                          :metamorph/data
                          (let [estimator (get ctx id)
                                attrs (set (dir estimator))

                                ]
                            (if (contains? attrs "predict")
                              (sklearn/predict data estimator)
                              (sklearn/transform data estimator {})
                              )


                            )
                          ))))


(defn fit-transform [module-kw estimator-class-kw kw-args]
  (fn [{:metamorph/keys [data id] :as ctx}]
    ;; (def data data)
    ;; (def id id)
    ;; (def ctx ctx)
    (let [{:keys [ds estimator]} (sklearn/fit-transform data module-kw estimator-class-kw kw-args )]
      (assoc ctx
             :metamorph/data ds
             id estimator))))
