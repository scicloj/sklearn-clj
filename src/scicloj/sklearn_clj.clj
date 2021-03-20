(ns scicloj.sklearn-clj
  (:require [camel-snake-kebab.core :as csk]
            [libpython-clj2.python.np-array]
            [libpython-clj2.metadata]
            [libpython-clj2.python :refer [->jvm as-jvm call-attr call-attr-kw cfn py.- py. python-type path->py-obj as-python]]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column :as ds-col]

            [tech.v3.dataset.tensor :as dst]
            [tech.v3.tensor :as t]))
(defn mapply [f & args] (apply f (apply concat (butlast args) (last args))))

(defn snakify-keys
  "Recursively transforms all map keys from to snake case."
  {:added "1.1"}
  [m]
  (let [f (fn [[k v]] (if (keyword?  k) [(csk/->snake_case_keyword k) v] [k v]))]
    ;; only apply to maps
    (clojure.walk/postwalk (fn [x] (if (map? x) (into {} (map f x)) x)) m)))


(defn- make-estimator [module-kw estimator-class kw-args]
  (let [
        snakified-kw-args (snakify-keys kw-args)
        module (csk/->snake_case_string module-kw)
        class-name (if (keyword? estimator-class)
                     (csk/->PascalCaseString estimator-class)
                     estimator-class)
        estimator-class-name (str "sklearn." module "." class-name)
        constructor (path->py-obj estimator-class-name )]
    (mapply cfn constructor snakified-kw-args))  )


(defn raw-tf-result->ds [raw-result]
  (let [raw-type (python-type raw-result)
        result (case raw-type
                 :csr-matrix (py. raw-result toarray)
                 :ndarray raw-result)
        new-ds
        (->  result as-jvm dst/tensor->dataset) ]
    new-ds))

(defn ds->X [ds]
  (let [
        string-ds (cf/of-datatype ds :string)
        X (if (nil? string-ds)
            (-> ds (dst/dataset->tensor) t/ensure-tensor as-python)
            (-> ds ds/columns first ))]
    X))


(defn fit
  "Call the fit method of a sklearn transformer, which is specified via
   the two keywords `module-kw` and `estimator-class-kw`.
  Keyword arguments can be given in a map `kw-arguments`.
  The data need to be given as tech.ml.dataset and will be converted to python automaticaly.
  The function will return the estimator as a python object.
  "
  [ds module-kw estimator-class-kw kw-args]
  (let
      [inference-targets (cf/target ds)
       feature-ds (cf/feature ds)
       estimator (make-estimator module-kw estimator-class-kw kw-args)
       X (ds->X feature-ds)]
      (if (nil? inference-targets)
        (py. estimator fit X)
        (let [y (-> inference-targets (dst/dataset->tensor) t/ensure-tensor as-python) ]
          (py. estimator fit X y)))))


  (defn fit-transform
    "Call the fit_transform method of a sklearn transformer, which is specified via
   the two keywords `module-kw` and `estimator-class-kw`.
  Keyword arguments can be given in a map `kw-arguments`.
  The data need to be given as tech.ml.dataset and will be converted to python automaticaly.
  The function will return the  estimator and transformed data as a tech.ml.dataset
  "
    [ds module-kw estimator-class-kw kw-args]
    (let
        [
         feature-ds (cf/feature ds)
         inference-target-ds (cf/target ds)
         estimator (make-estimator module-kw estimator-class-kw kw-args)
         X (ds->X feature-ds)
         raw-result (py. estimator fit_transform X)

         new-ds (raw-tf-result->ds raw-result)
         new-ds  (if (nil? inference-target-ds)
                   new-ds
                   (ds/append-columns new-ds (ds/columns inference-target-ds)))]
        {:ds new-ds
         :estimator estimator}))


  (defn predict
    "Calls `predict` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
    [ds estimator]
    (let
        [
         feature-ds (cf/feature ds)
         inference-target-column-names (ds-mod/inference-target-column-names ds)
         X  (ds->X feature-ds)
         prediction (py. estimator predict X)
         y_hat-ds
         (->>
          prediction
          as-jvm
          (ds-col/new-column (first inference-target-column-names))
          vector
          (ds/new-dataset )
          )
         ]
        (ds/append-columns feature-ds (ds/columns y_hat-ds))))

;; (-> prediction as-jvm (ds-col/new-column :x))
(defn transform
  "Calls `transform` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  [ds estimator kw-args]
  (let [snakified-kw-args (snakify-keys kw-args)
        feature-ds (cf/feature ds)
        column-names (ds/column-names feature-ds)
        X (ds->X feature-ds)
        raw-result (py. estimator transform X )
        X-transformed (raw-tf-result->ds raw-result )
        target-ds (cf/target ds)
        ]
    (if (nil? target-ds)
      X-transformed
      (ds/concat
       X-transformed
       (cf/target ds)))))


(comment
  (def est
    (make-estimator :svm "SVC" {}))

  (contains?

   (libpython-clj2.python/dir est)
   "predict")
  )
