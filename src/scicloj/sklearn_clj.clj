(ns scicloj.sklearn-clj
  (:require [camel-snake-kebab.core :as csk]
            libpython-clj.metadata
            [libpython-clj.python :refer [->jvm ->numpy call-kw py.]]
            [tablecloth.api :as tc]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.tensor :as dst]
            [tech.v3.tensor :as t]))

(defn snakify-keys
  "Recursively transforms all map keys from to snake case."
  {:added "1.1"}
  [m]
  (let [f (fn [[k v]] (if (keyword?  k) [(csk/->snake_case k) v] [k v]))]
    ;; only apply to maps
    (clojure.walk/postwalk (fn [x] (if (map? x) (into {} (map f x)) x)) m)))



(defn fit-transform
  [ds module-kw estimator-class-kw kw-args
   ]
  (def ds ds)
  (user/def-let
    [inference-targets (cf/target ds)
     feature-ds (cf/feature ds)
     snakified-kw-args (snakify-keys kw-args)
     module (csk/->snake_case_string module-kw)
     class-name (csk/->PascalCaseString estimator-class-kw)
     estimator-class-name (str "sklearn." module "." class-name)
     constructor (libpython-clj.metadata/path->py-obj estimator-class-name )
     estimator (call-kw constructor [] kw-args)
     X (-> feature-ds (dst/dataset->tensor) ->numpy)]
    (->
     (py. estimator fit_transform X )
     (t/ensure-tensor)
     ;; ->jvm
     (dst/tensor->dataset)
     (tc/append inference-targets)
     )
    ))



(defn fit
  [ds module-kw estimator-class-kw kw-args
   ]
  (let
    [inference-targets (cf/target ds)
     feature-ds (cf/feature ds)
     snakified-kw-args (snakify-keys kw-args)
     module (csk/->snake_case_string module-kw)
     class-name (csk/->PascalCaseString estimator-class-kw)
     estimator-class-name (str "sklearn." module "." class-name)
     constructor (libpython-clj.metadata/path->py-obj estimator-class-name )
     estimator (call-kw constructor [] kw-args)
     X (-> feature-ds (dst/dataset->tensor) ->numpy)
     y (-> inference-targets (dst/dataset->tensor) ->numpy)]
    (py. estimator fit X y)
    ))

(defn predict
  [ds estimator kw-args]
  (let
    [
     feature-ds (cf/feature ds)
     snakified-kw-args (snakify-keys kw-args)
     X (-> feature-ds (dst/dataset->tensor) ->numpy)
     y_hat
     (->
      (py. estimator predict X)
      (t/ensure-tensor)
      ->jvm
      (dst/tensor->dataset)
      )]
    (tc/append feature-ds y_hat)


    ))




(defn transform
  [ds estimator kw-args]
  (let [
        snakified-kw-args (snakify-keys kw-args)

     feature-ds (cf/feature ds)
        X (-> feature-ds (dst/dataset->tensor) ->numpy)]
    (->
     (py. estimator transform X )
      (t/ensure-tensor)
      ->jvm
      (dst/tensor->dataset)
     )
    ))





(comment
  (def train-ds
    (->
     (tc/dataset {:x1 [1 2 3]
                  :x2  [7 8 9]
                  :y [10 10 10 ]})
     (ds-mod/set-inference-target :y)
     ))

  (def test-ds (->
                (tc/dataset {:x1 [0 0 0]
                             :x2  [2 4 6]
                             :y [0 0 0 ]})
                (ds-mod/set-inference-target :y)
                ))

  (def lin-reg
    (fit train-ds :linear_model :linear-regression ;; {:y [ 9 8 7]}
         {}
         ))
  (predict test-ds lin-reg {})




  (def scaler
    (fit train-ds :preprocessing :standard-scaler {})

    )

  (transform test-ds scaler {})

  )
