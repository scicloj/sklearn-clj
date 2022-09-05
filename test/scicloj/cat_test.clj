(ns scicloj.cat-test
  (:require
   [taoensso.nippy :as nippy]
   [libpython-clj2.python :as py]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.metamorph :as ds-mm]
   [scicloj.metamorph.core :as morph]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.categorical :as ds-cat]

   [tech.v3.dataset.column-filters :as cf]
   [tablecloth.api.split :as split]
   [scicloj.metamorph.ml :as ml]
   [scicloj.metamorph.ml.loss :as loss]
   [scicloj.sklearn-clj.ml]
   [clojure.test :refer [deftest is]]))
   

(def builtins (py/import-module "builtins"))


(def pipe-fn (morph/pipeline
              (ds-mm/set-inference-target :species)
              (ds-mm/categorical->number cf/categorical)
              {:metamorph/id :model}
              (ml/model {:model-type
                         :sklearn.classification/random-forest-classifier})))
(def ds (ds/->dataset "https://raw.githubusercontent.com/techascent/tech.ml/master/test/data/iris.csv" {:key-fn keyword}))


(deftest test-categorical
  (let [
        train-split-seq (split/split->seq ds :holdout)
        pipe-fn-seq [pipe-fn]
        evaluations (ml/evaluate-pipelines pipe-fn-seq train-split-seq loss/classification-loss :loss)
        best-fitted-context (-> evaluations first first :fit-ctx)
        _ (def best-fitted-context best-fitted-context)
        best-pipe-fn (-> evaluations first first :pipe-fn)
        new-ds (ds/sample ds 10 {:seed 1234})
        predictions
        (->
         (best-pipe-fn
          (merge best-fitted-context
                 {:metamorph/data new-ds
                  :metamorph/mode :transform}))
         (:metamorph/data)
         (ds-mod/column-values->categorical :species))]

         
    (is (=  {"versicolor" 5, "virginica" 4, "setosa" 1}
            (frequencies predictions)))))



(deftest serialize-ctx
  (let [ctx
        (morph/fit-pipe ds pipe-fn)
        _ (nippy/freeze-to-file "/tmp/ctx.nippy" ctx)
        loaded-ctx (nippy/thaw-from-file "/tmp/ctx.nippy")]
    (is (= {"setosa" 50, "versicolor" 50, "virginica" 50}
           (->
            (morph/transform-pipe ds
                                  pipe-fn
                                  loaded-ctx)
            :metamorph/data
            ds-cat/reverse-map-categorical-xforms
            :species
            frequencies)))))











(comment


  (-> (ds/->dataset  {:x [:a :b]})
      (ds/categorical->number  cf/categorical [[:a 1 :b 2]])
      (-> :x meta :categorical-map :lookup-table))




  (-> (ds/->dataset  {:x [:a :b]})
      (ds/categorical->number  cf/categorical [[:a 3.1 ] [:b 3.2]] :int16)
      (ds-cat/reverse-map-categorical-xforms))


  (-> (ds/->dataset  {:x [:a :b]})
      (ds/categorical->number  cf/categorical [[:b 3.1 ] [:a 7.1]])
      (ds-cat/reverse-map-categorical-xforms)))
