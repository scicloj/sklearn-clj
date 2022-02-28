(ns scicloj.sklearn-clj-test
  (:require [clojure.test :refer :all]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.modelling :as ds-mod]
            [tablecloth.api :as tc]
            [tech.v3.dataset.tensor :as dst]
            [tech.v3.tensor :as tensor]
            [libpython-clj2.python :refer [python-type] :as py]
            [tech.v3.datatype.functional :as f]
            [scicloj.metamorph.ml.toydata :as toydata]
            [scicloj.metamorph.ml :as ml]
            [scicloj.metamorph.core :as mm]
            [scicloj.sklearn-clj :refer :all]
            [tablecloth.pipeline :as tcm]
            [tech.v3.dataset.metamorph :as ds-mm]
            [tech.v3.dataset.column-filters :as cf]
            [scicloj.sklearn-clj.ml]))

            

(def texts ["This is the first document."
               "This document is the second document."
               "And this is the third one."
               "Is this the first document?"])
            

(defn values->ds [values shape datatype]
  (->
   (tensor/->tensor values :datatype datatype)
   (tensor/reshape shape)
   (dst/tensor->dataset)))
  



(deftest count-vectorizer-fit-transform-with-target
 (let [ds (->
             (ds/->dataset {:text texts
                            :target (repeat (count texts) 5)})
             (ds-mod/set-inference-target :target))
         result
         (-> ds
             (fit-transform :sklearn.feature-extraction.text :Count-Vectorizer {})
             :ds dst/dataset->tensor)]
     (is (f/equals result
                   (tensor/->tensor
                    [[0 1 1 1 0 0 1 0 1 5]
                     [0 2 0 1 0 1 1 0 1 5]
                     [1 0 0 1 1 0 1 1 1 5]
                     [0 1 1 1 0 0 1 0 1 5]]
                    :datatype :float64)))))



(deftest count-vectorizer-fit

  (let [ds (ds/->dataset {:text texts})
        estimator (fit ds :sklearn.feature-extraction.text :Count-Vectorizer {})
        prediction (transform ds estimator {})]
        
    (is (= :count-vectorizer
           (python-type estimator)))))
    

(deftest tfidf-vectorizer

  (let [ds (ds/->dataset {:text texts})
        result
        (-> ds
            (fit-transform :sklearn.feature-extraction.text :TfidfVectorizer {})
            :ds
            dst/dataset->tensor)]
            
    (is (f/equals result

                  (tensor/->tensor
                   [[ 0.000 0.4698 0.5803 0.3841  0.000  0.000 0.3841  0.000 0.3841]
                    [ 0.000 0.6876  0.000 0.2811  0.000 0.5386 0.2811  0.000 0.2811]
                    [0.5118  0.000  0.000 0.2671 0.5118  0.000 0.2671 0.5118 0.2671]
                    [ 0.000 0.4698 0.5803 0.3841  0.000  0.000 0.3841  0.000 0.3841]]
                   :datatype :float64)))))

(deftest normalizer
  (is (f/equals
       (-> [[4 1 2 2]
            [1 3 9 3]
            [5 7 5 1]]
           (tensor/->tensor :datatype :float64)
           (dst/tensor->dataset)
           (fit-transform :sklearn.preprocessing :normalizer {})
           :ds
           dst/dataset->tensor)
           
       (tensor/->tensor
        [[0.8 0.2 0.4 0.4]
         [0.1 0.3 0.9 0.3],
         [0.5 0.7 0.5 0.1]]
        :datatype :float64))))

(deftest normalizer-single-row
  (is (f/equals
       (-> [[4 1 2 2]]
           (tensor/->tensor :datatype :float64)
           (dst/tensor->dataset)
           (fit-transform :sklearn.preprocessing :normalizer {})
           :ds
           dst/dataset->tensor)
           
       (tensor/->tensor
        [[0.8 0.2 0.4 0.4]]
        :datatype :float64))))



(deftest normalizer-single-col
  (is (f/equals
       (-> [[4]
            [1]
            [5]]
           (tensor/->tensor :datatype :float64)
           (dst/tensor->dataset)
           (fit-transform :sklearn.preprocessing :normalizer {})
           :ds
           dst/dataset->tensor)
           
       (tensor/->tensor
        [[1]
         [1],
         [1]]
        :datatype :float64))))


(def n-arr-1
  (values->ds [32 nil 28 nil 32 nil nil 34 40]
              [3 3]
              :object))
              

(def n-arr-2
  (values->ds [6 9 nil 7 nil nil nil 8 12]
              [3 3]
              :object))
              



(deftest impute-1
  (let [imp (fit n-arr-1 :sklearn.impute :simple-imputer)
        imputed (transform n-arr-2 imp  {})]
    (is (=
         (values->ds
          [6 9 34
           7 33 34
           32 8 12]
          [3 3]
          :float64)
          
         imputed))))
         
    

(deftest impute-2
  (let [{:keys [estimator ds]} (fit-transform n-arr-2 :sklearn.impute :simple-imputer)]
        ;; imputed (transform n-arr-2 imp  {})
    (is (=
         (values->ds
          [6    9   12
           7    8.5 12
           6.5  8   12]
          [3 3]
          :float64)
          
         ds))))
         
    
(deftest impute-3
  (let [estimator (fit n-arr-2 :sklearn.impute :simple-imputer)
        imputed (transform n-arr-2 estimator {})]
        ;; imputed (transform n-arr-2 imp  {})
        
    (is (=
         (values->ds
          [6    9   12
           7    8.5 12
           6.5  8   12]
          [3 3]
          :float64)
          
         imputed))))
         
    



(deftest model-svc

  (let [iris-split
        (->
         (toydata/iris-ds)
         (tc/split->seq :holdout {:seed 1234})
         first)

        pipe-fn
        (mm/pipeline
         {:metamorph/id :svc}
         (ml/model {:model-type :sklearn.classification/svc}))

        fitted-ctx
        (mm/fit-pipe
         (:train iris-split)
         pipe-fn)]


    (is (=  {1.0 18, 0.0 15, 2.0 17}

            (->
             (mm/transform-pipe
              (:test iris-split)
              pipe-fn
              fitted-ctx)
             :metamorph/data
             :species
             frequencies)))))
