(ns scicloj.sklearn-clj.metamorph
  (:require [tech.v3.dataset.modelling :as ds-mod]
            [scicloj.sklearn-clj :as sklearn]

            [libpython-clj2.python :refer [dir ->jvm  py.- py. python-type]]))
            

(defn estimate
  ([module-kw estimator-class-kw]
   (estimate module-kw estimator-class-kw {}))
  ([module-kw estimator-class-kw kw-args]
   (fn [{:metamorph/keys [id data mode] :as ctx}]
     (def data data)
     (case mode

       :fit (let [estimator (sklearn/fit data module-kw estimator-class-kw kw-args)]

              (assoc ctx
                     id
                     {:estimator estimator
                      :attributes (sklearn/model-attributes estimator)}))
       :transform (assoc ctx
                         :metamorph/data
                         (let [estimator (get-in ctx [id :estimator])
                               attrs (set (dir estimator))]
                           (if (contains? attrs "predict")
                             (sklearn/predict data estimator)
                             (sklearn/transform data estimator kw-args))))))))



                            
                          


(defn fit-transform
  ([module-kw estimator-class-kw]
   (fit-transform module-kw estimator-class-kw {}))
  ([module-kw estimator-class-kw kw-args]
   (fn [{:metamorph/keys [data id] :as ctx}]
     (let [{:keys [ds estimator]} (sklearn/fit-transform data module-kw estimator-class-kw kw-args)]
       (assoc ctx
              :metamorph/data ds
              id estimator)))))
