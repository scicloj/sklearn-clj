(ns scicloj.sklearn-clj
  (:require [camel-snake-kebab.core :as csk]
            [libpython-clj2.python.np-array]
            [libpython-clj2.metadata]
            [libpython-clj2.python :refer [->jvm as-jvm call-attr
                                           call-attr-kw cfn py.- py.
                                           python-type path->py-obj as-python] :as py]

            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.column :as ds-col]
            [tech.v3.datatype.errors :as errors]

            [tech.v3.dataset.tensor :as dst]
            [tech.v3.tensor :as t]))

(defmacro when-error
  "Throw an error in the case where expr is true."
  [expr error-msg]
  {:style/indent 1}
  `(when ~expr
     (throw (Exception. ~error-msg))))
(defmacro xor
  "Evaluates exprs one at a time, from left to right.  If only one form returns
  a logical true value (neither nil nor false), returns true.  If more than one
  value returns logical true or no value returns logical true, retuns a logical
  false value.  As soon as two logically true forms are encountered, no
  remaining expression is evaluated.  (xor) returns nil."
  ([] nil)
  ([f & r]
   `(loop [t# false f# '[~f ~@r]]
      (if-not (seq f#) t#
              (let [fv# (eval (first f#))]
                (cond
                  (and t# fv#) false
                  (and (not t#) fv#) (recur true (rest f#))
                  :else (recur t# (rest f#))))))))

(defn mapply [f & args] (apply f (apply concat (butlast args) (last args))))

(defn snakify-keys
  "Recursively transforms all map keys from to snake case."
  {:added "1.1"}
  [m]
  (let [f (fn [[k v]] (if (keyword?  k) [(csk/->snake_case_keyword k) v] [k v]))]
    ;; only apply to maps
    (clojure.walk/postwalk (fn [x] (if (map? x) (into {} (map f x)) x)) m)))


(defn make-estimator [module-kw estimator-class kw-args]
  (let [
        snakified-kw-args (snakify-keys kw-args)
        module (csk/->snake_case_string module-kw)
        class-name (if (keyword? estimator-class)
                     (csk/->PascalCaseString estimator-class)
                     estimator-class)
        estimator-class-name (str module "." class-name)
        constructor (path->py-obj estimator-class-name)]
    (mapply cfn constructor snakified-kw-args)))


(defn raw-tf-result->ds [raw-result]
  (let [raw-type (python-type raw-result)
        result (case raw-type
                 :csr-matrix (py. raw-result toarray)
                 :ndarray raw-result)
        new-ds
        (->  result as-jvm dst/tensor->dataset)]
    new-ds))

(defn ds->X [ds]
  (let [
        string-ds (cf/of-datatype ds :string)
        numeric-ds (cf/numeric ds)
        _ (when-error  (and (some? string-ds) (some? numeric-ds))
            "Dataset contains numeric and non-numeric features, which is not supported.")

        X (if (nil? string-ds)
            (-> ds (dst/dataset->tensor) t/ensure-tensor as-python)

            (do
              (errors/when-not-error (= 1 (ds/column-count string-ds))
                                     "Dataset contains more then 1 string column, which is not supported.")
                                     
              (-> ds ds/columns first)))]
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
      [feature-ds (cf/feature ds)
       X (ds->X feature-ds)
       estimator (make-estimator module-kw estimator-class-kw (dissoc kw-args :unsupervised?))]

      (if (kw-args :unsupervised?)
        (py. estimator fit X)
        (let [
              inference-targets (cf/target ds)
              inference-targets (cf/numeric inference-targets)
              _ (errors/when-not-error inference-targets "No numeric inference targets found in dataset.")
              y (-> inference-targets (dst/dataset->tensor) t/ensure-tensor as-python)]
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
       estimator (make-estimator module-kw estimator-class-kw (dissoc  kw-args :unsupervised?))
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
  ([ds estimator inference-target-column-names]
   (let
       [feature-ds (cf/feature ds)
        X  (ds->X feature-ds)
        prediction (py. estimator predict X)

        y_hat-ds
        (->>
         prediction
         as-jvm
         (ds-col/new-column (first inference-target-column-names))
         vector
         (ds/new-dataset))]
           
          
     (ds/append-columns feature-ds (ds/columns y_hat-ds))))
  ([ds estimator]
   (predict ds estimator (ds-mod/inference-target-column-names ds))))
    

;; (-> prediction as-jvm (ds-col/new-column :x))
(defn transform
  "Calls `transform` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  [ds estimator kw-args]
  (let [snakified-kw-args (snakify-keys kw-args)
        feature-ds (cf/feature ds)
        column-names (ds/column-names feature-ds)
        X (ds->X feature-ds)
        raw-result (py. estimator transform X)
        X-transformed (raw-tf-result->ds raw-result)
        target-ds (cf/target ds)]
        
    (if (nil? target-ds)
      X-transformed
      (ds/append-columns
       X-transformed
       (cf/target ds)))))


(defn model-attribute-names [sklearn-model]

  (->> (py/dir sklearn-model)
   (filter #(and  (clojure.string/ends-with? % "_")
               (not (clojure.string/starts-with? % "_"))))))

(defn model-attributes [sklearn-model]
  (map
   (fn [attr]
     (hash-map (keyword attr)
               (py/->jvm (py/get-attr sklearn-model attr))))
   (model-attribute-names sklearn-model)))


(comment
  (make-estimator :umap "UMAP" {})
  (make-estimator :sklearn.svm "SVC" {})
  (get-model-attributes
   (make-estimator :sklearn.svm "SVC" {}))
  (def est)



  (contains?

   (libpython-clj2.python/dir est)
   "predict"))
  
