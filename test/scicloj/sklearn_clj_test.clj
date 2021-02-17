(ns scicloj.sklearn-clj-test
  (:require [clojure.test :refer :all]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.tensor :as dst]
            [tech.v3.tensor :as tensor]
            [tech.v3.datatype.functional :as f]
            [scicloj.sklearn-clj :refer :all]))

(def texts ["This is the first document."
               "This document is the second document."
               "And this is the third one."
               "Is this the first document?"
            ])


(deftest count-vectorizer

  (let [ds (ds/->dataset {:text texts})
        result
        (-> ds
            (fit-transform :feature-extraction.text :Count-Vectorizer {})
            :ds
            dst/dataset->tensor
            )]
    (is (f/equals result
                  (tensor/->tensor
                   [[0 1 1 1 0 0 1 0 1]
                    [0 2 0 1 0 1 1 0 1]
                    [1 0 0 1 1 0 1 1 1]
                    [0 1 1 1 0 0 1 0 1]]
                    :datatype :float64)))))

(deftest tfidf-vectorizer

  (let [ds (ds/->dataset {:text texts})
        result
        (-> ds
            (fit-transform :feature-extraction.text :TfidfVectorizer {})
            :ds
            dst/dataset->tensor
            )]
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
           (fit-transform :preprocessing :normalizer {})
           :ds
           dst/dataset->tensor
           )
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
           (fit-transform :preprocessing :normalizer {})
           :ds
           dst/dataset->tensor
           )
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
           (fit-transform :preprocessing :normalizer {})
           :ds
           dst/dataset->tensor
           )
       (tensor/->tensor
        [[1]
         [1],
         [1]]
        :datatype :float64))))
