(ns scicloj.sklearn-wrapper-test
  (:require
   [clojure.test :as t]
   [libpython-clj2.python :as py]
   [scicloj.metamorph.core :as mm]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.toydata :as toydata]
   [scicloj.sklearn-clj.sklearn-wrapper]
   [tablecloth.api :as tc]
   [scicloj.sklearn-clj]
   [scicloj.sklearn-clj.ml]
   [scicloj.ml.smile.classification]
   [tech.v3.dataset.categorical :as ds-cat]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.metamorph :as ds-mm]))

(py/initialize!)
(py/from-import yellowbrick.model_selection LearningCurve)

(defn fit-lc [estimator X y]
  (let [lc (LearningCurve estimator
                          :train_sizes [60 70 80 90 100 110 120])
        _ (py/py. lc fit X y)]
    lc))



(t/deftest wrap-via-lc

  (let [data
        (-> (toydata/iris-ds)
            (tc/shuffle {:seed 12345}))
        ;; (tc/select-columns [:sepal_length :sepal_wi:species])


        _ (def data data)
        pipe-fn
        (mm/pipeline
         (ds-mm/set-inference-target :species)
         (ml/model {:model-type :smile.classification/random-forest}))

        X (-> data
              (cf/feature)
              (scicloj.sklearn-clj/ds->X))


        y (-> data
              (cf/target)
              (scicloj.sklearn-clj/ds->X))

        X-column-names
        (-> data
            (cf/feature)
            (tc/column-names))

        X-categorical-maps
        (-> data
            (cf/feature)
            (ds-cat/dataset->categorical-maps))


        y-column-names
        (-> data
            (cf/target)
            (tc/column-names))


        y-categorical-maps
        (-> data
            (cf/target)
            (ds-cat/dataset->categorical-maps)
            first)


        _ (def pipe-fn pipe-fn)
        _ (def X-column-names X-column-names)
        _ (def y-column-names y-column-names)
        _ (def X-categorical-maps X-categorical-maps )
        _ (def y-categorical-maps y-categorical-maps)
        estimator
        (scicloj.sklearn-clj.sklearn-wrapper/pipe-fn->estimator
         "my" pipe-fn
         X-column-names y-column-names
         X-categorical-maps y-categorical-maps)

        _ (def estimator estimator)
        _ (def X X)
        _ (def y y)
        lc
        (fit-lc estimator X y)]


    (def lc lc)
    (t/is (= [60 70 80 90 100 110 120]
             (->
              (py/py.- lc train_sizes_)
              (py/->jvm)
              vec)))))



