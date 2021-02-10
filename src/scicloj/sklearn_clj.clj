(ns scicloj.sklearn-clj
  (:require [camel-snake-kebab.core :as csk]
            libpython-clj.metadata
            [libpython-clj.python :refer [->jvm ->numpy call-kw py.- py.]]
            [tech.v3.dataset :as ds]
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




(defn fit
  "Call the fit method of a sklearn transformer, which is specified via
   the two keywords `module-kw` and `estimator-class-kw`.
  Keyword arguments can be given in a map `kw-arguments`.
  The data need to be given as tech.ml.dataset and will be converted to pythjon automaticaly.
  The function will return the estimator as a python object.
  "
  [ds module-kw estimator-class-kw kw-args]
   (let
      [inference-targets (cf/target ds)
       feature-ds (cf/feature ds)
       snakified-kw-args (snakify-keys kw-args)
       module (csk/->snake_case_string module-kw)
       class-name (csk/->PascalCaseString estimator-class-kw)
       estimator-class-name (str "sklearn." module "." class-name)
       constructor (libpython-clj.metadata/path->py-obj estimator-class-name )
       estimator (call-kw constructor [] kw-args)
       X (-> feature-ds (dst/dataset->tensor) ->numpy)]
      (if (nil? inference-targets)
        (py. estimator fit X)
        (let [y (-> inference-targets (dst/dataset->tensor) ->numpy) ]
          (py. estimator fit X y)))))

(defn predict
  "Calls `predict` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  [ds estimator kw-args]
  (let
      [
       feature-ds (cf/feature ds)
       inference-target-column-names (ds-mod/inference-target-column-names ds)
       snakified-kw-args (snakify-keys kw-args)
       X (-> feature-ds (dst/dataset->tensor) ->numpy)
       y_hat
       (->
        (py. estimator predict X)
        (t/ensure-tensor)
        ->jvm
        (dst/tensor->dataset))
       y_hat (ds/rename-columns y_hat (zipmap (ds/column-names y_hat) inference-target-column-names))]
    (ds/append-columns feature-ds (ds/columns y_hat))))

(defn transform
  "Calls `transform` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  [ds estimator kw-args]
  (let [snakified-kw-args (snakify-keys kw-args)
        feature-ds (cf/feature ds)
        column-names (ds/column-names feature-ds)
        X (-> feature-ds (dst/dataset->tensor) ->numpy)
        X-transformed
        (-> (py. estimator transform X )
            (t/ensure-tensor)
            ->jvm
         (dst/tensor->dataset))]
    (ds/rename-columns
     X-transformed
     (zipmap
      (ds/column-names X-transformed)
      column-names))))





(comment

  (require
   '[libpython-clj.python :refer [py.-]]
   '[tech.v3.dataset :as ds]
   '[tech.v3.dataset.modelling :as ds-mod])

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

  (def lin-reg
    (fit train-ds :linear-model :linear-regression {}))

  (predict test-ds lin-reg {})
  ;; => _unnamed [1 3]:
  ;;    | :x1 | :x2 |   :y |
  ;;    |-----|-----|------|
  ;;    |   3 |   5 | 16.0 |


  (def data
    (ds/->dataset {:x1 [0 0 1 1]
                 :x2 [0 0 1 1]}))
  (def scaler
    (fit data :preprocessing :standard-scaler {}))

  (py.- scaler mean_)
  ;; => [0.5 0.5]
  ;;
  (transform (ds/->dataset {:x1 [2] :x2 [2]})  scaler {})
  ;; => :_unnamed [1 2]:
  ;;    | :x1 | :x2 |
  ;;    |-----|-----|
  ;;    | 3.0 | 3.0 |


  )
