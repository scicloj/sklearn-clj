[![Clojars Project](https://img.shields.io/clojars/v/scicloj/sklearn-clj.svg)](https://clojars.org/scicloj/sklearn-clj)

# sklearn-clj

This library gives easy access in Clojure to all estimators and models from python [scikit-learn,](https://scikit-learn.org/)
using internally libpython-clj.


It uses a tech.ml.dataset for input and output and converts this data structures to and from python as needed. This "should" be zero copy now.

As all estimators and models in sklearn uses the same interface,  it works for all estimators.


## Usage

Setup libpython-clj including sklearn.

If you need to call `py/initialize!` as part of your libpython-clj setup, this needs to happen before require
the other namesspaces. This can be done easely by adding a `user.clj` file which contains the call to `py/initialize!`  

See here https://github.com/clj-python/libpython-clj#usage for more information how to setup libpython-clj.



```clojure

 
  (require
   '[libpython-clj2.python :refer [py.-]]
   '[tech.v3.dataset :as ds]
   '[tech.v3.dataset.modelling :as ds-mod]
   '[scicloj.sklearn-clj :refer :all]
   )

;; example train and test dataset 

  (def train-ds
    (-> (ds/->dataset {:x1 [1 1 2 2]
                  :x2 [1 2 2 3]
                  :y  [6 8 9 11]})
     (ds-mod/set-inference-target :y)))

  (def test-ds
    (->
     (ds/->dataset {:x1 [3]
                  :x2 [5]
                  :y  [0]})
     (ds-mod/set-inference-target :y)))

;; fit a liner expression model from sklearn, class sklearn.linear_model.LinearRegression
 
  (def lin-reg
    (fit train-ds :linear-model :linear-regression ))

;; Call predict with new data on the estimator
  (predict test-ds lin-reg {})
  ;; => _unnamed [1 3]:
  ;;    | :x1 | :x2 |   :y |
  ;;    |-----|-----|------|
  ;;    |   3 |   5 | 16.0 |

;; use an other estimator, this time: sklearn.preprocessing.StandardScaler
  (def data
    (ds/->dataset {:x1 [0 0 1 1]
                 :x2 [0 0 1 1]}))
                 
;; fit the scaler on data                 
  (def scaler
    (fit data :preprocessing :standard-scaler))

  (py.- scaler mean_)
  ;; => [0.5 0.5]
  ;;
  
;; apply the scaling on new data  
  (transform (ds/->dataset {:x1 [2] :x2 [2]})  scaler)
  ;; => :_unnamed [1 2]:
  ;;    | :x1 | :x2 |
  ;;    |-----|-----|
  ;;    | 3.0 | 3.0 |

```

The library provides as well an adaptor to the scicloj [metamorph](https://github.com/scicloj/metamorph)  library, in order to use the estimators inside a metamorph pipeline.


```clojure
(require '[scicloj.sklearn-clj.metamorph :as sklearn-mm])

;; adding this as a pipeline operation somewhere in the pipeline
;; The estimate functon will do the right thing, depending on the :metamorph/mode   key being
;; :fit or :transform  , namely calling "fit" on :fit  and "predict" on :transform
(sklearn-mm/estimate :linear-model :linear-regression {})
```


Alternatively the models can be integrated in tech.ml / [scicloj.ml](https://github.com/scicloj/scicloj.ml)

``` clojure
(require '[scicloj.sklearn-clj.ml]) ;; registers all models
(require '[scicloj.ml.core :as ml]
         '[scicloj.ml.metamorph :as mm]
         '[scicloj.ml.dataset :as ds]
         '[libpython-clj2.python :refer [py. py.-]])

(def iris
  (->
   (ds/dataset
    "https://raw.githubusercontent.com/scicloj/metamorph.ml/main/test/data/iris.csv" {:key-fn keyword})))


(def pipe-fn   
 (ml/pipeline
  (mm/set-inference-target :species)
  (mm/categorical->number [:species])
  {:metamorph/id :model}
  (mm/model {:model-type :sklearn.classification/logistic-regression})))

(def trained-ctx (ml/fit-pipe iris pipe-fn))

(def model-object
  (-> trained-ctx :model :model-data :model))


(py.- model-object coef_)
;; => [[ 0.53399346 -0.31779319 -0.20533681 -0.93955019]
;;     [-0.42332856  0.96165291 -2.51935878 -1.08617922]
;;     [-0.1106649  -0.64385972  2.72469559  2.02572941]]

```

All available models with their key, options and complete documentation are listed here:



https://scicloj.github.io/scicloj.ml-tutorials/userguide-sklearnclj.html

## License

Copyright © 2021 Scicloj

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
