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


(def data (-> (toydata/iris-ds)
            (tc/shuffle {:seed 12345})))


(t/deftest wrap-via-lc

  (let [data
        (-> (toydata/iris-ds)
            (tc/shuffle {:seed 12345}))
        ;; (tc/select-columns [:sepal_length :sepal_wi:species])


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


        estimator
        (scicloj.sklearn-clj.sklearn-wrapper/pipe-fn->estimator
         "my" pipe-fn
         X-column-names y-column-names
         X-categorical-maps y-categorical-maps)

        _ (def X X)
        _ (def y y)
        _ (def estimator estimator)
        lc
        (fit-lc estimator X y)]


    (t/is (= [60 70 80 90 100 110 120]
             (->
              (py/py.- lc train_sizes_)
              (py/->jvm)
              vec)))))



(comment

  (defn fit-and-show-lc! [estimator X y]
    (let [lc (LearningCurve estimator
                            ;; :cv 5
                            ;; :scoring "f1_micro"
                            :train_sizes [30 40 50 60 70 80 90 100 110 120])


          _ (py/py. lc fit X y)]
     (py/py. lc show)

     lc)))

(comment
  (def
    X (-> data
          (cf/feature)
          (scicloj.sklearn-clj/ds->X)))

  (def
    y (-> data
          (cf/target)
          (scicloj.sklearn-clj/ds->X)
          (py/py. reshape [150]))))



 
(py/py.- y shape)


(first)
(comment
  (py/from-import  yellowbrick.model_selection  CVScores)


  (def vis (CVScores estimator :cv 20))
  (py/py. vis fit X y)
  (py/py. vis show)


  (comment
    (py/from-import  yellowbrick.target FeatureCorrelation)

    (def vis (FeatureCorrelation :labels (-> data cf/feature tc/column-names vec)))
    (py/py. vis fit X y)
    (py/py. vis show)



    :ok)


  :ok)
