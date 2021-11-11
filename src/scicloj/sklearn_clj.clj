(ns scicloj.sklearn-clj
  (:require
   [camel-snake-kebab.core :as csk]
   [libpython-clj2.python
    :as py
    :refer [as-jvm as-python cfn path->py-obj python-type py. py.-]]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.column-filters :as cf]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.tensor :as dst]
   [tech.v3.datatype.errors :as errors]
   [tech.v3.tensor :as t]
   [libpython-clj2.python.np-array]))

(println  "'sklearn' version found: "
          (get
           (py/module-dict (py/import-module "sklearn"))
           "__version__"))

(defmacro when-error
  "Throw an error in the case where expr is true."
  [expr error-msg]
  {:style/indent 1}
  `(when ~expr
     (throw (Exception. ~error-msg))))



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
  (def raw-result raw-result)
  (let [raw-type (python-type raw-result)
        result (case raw-type
                 :csr-matrix (py. raw-result toarray)
                 :ndarray raw-result)
        new-ds
        (->  result py/->jvm dst/tensor->dataset)]
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


(defn prepare-python [ds module-kw estimator-class-kw kw-args]
  {:estimator (make-estimator module-kw estimator-class-kw kw-args)
   :inference-target (cf/target ds)
   :X (-> ds cf/feature ds->X)
   :y (some->  ds cf/target dst/dataset->tensor t/ensure-tensor as-python)})


(defn fit
  "Call the fit method of a sklearn transformer, which is specified via
   the two keywords `module-kw` and `estimator-class-kw`.
  Keyword arguments can be given in a map `kw-arguments`.
  The data need to be given as tech.ml.dataset and will be converted to python automaticaly.
  The function will return the estimator as a python object.
  "
  ([ds module-kw estimator-class-kw]
   (fit ds module-kw estimator-class-kw {}))
  ([ds module-kw estimator-class-kw kw-args]
   (let [{:keys [estimator X y]} (prepare-python ds module-kw estimator-class-kw kw-args)]
     (def X_1 X)
     (if y
       (py. estimator fit X y)
       (py. estimator fit X)))))

(defn fit-transform
  "Call the fit_transform method of a sklearn transformer, which is specified via
   the two keywords `module-kw` and `estimator-class-kw`.
  Keyword arguments can be given in a map `kw-arguments`.
  The data need to be given as tech.ml.dataset and will be converted to python automaticaly.
  The function will return the  estimator and transformed data as a tech.ml.dataset
  "
  ([ds module-kw estimator-class-kw]
   (fit-transform ds module-kw estimator-class-kw {}))
  ([ds module-kw estimator-class-kw kw-args]

   (let [{:keys [estimator X y inference-target]} (prepare-python ds module-kw estimator-class-kw kw-args)
         raw-result (py. estimator fit_transform X)
         new-ds (raw-tf-result->ds raw-result)
         new-ds (if (nil? inference-target)
                  new-ds
                  (ds/append-columns new-ds (ds/columns inference-target)))]
     {:ds new-ds
      :estimator estimator})))


(defn predict
  "Calls `predict` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  ([ds estimator inference-target-column-names]
   (let
       [feature-ds (cf/feature ds)
        _ (def feature-ds feature-ds)
        X  (ds->X feature-ds)
        _ (def X X)
        _ (def estimator estimator)
        prediction (py. estimator predict X)
        _ (def prediction prediction)
        _ (def inference-target-column-names inference-target-column-names)
        y_hat-ds (ds/->dataset {(first inference-target-column-names) prediction})]
      (ds/append-columns feature-ds (ds/columns y_hat-ds))))
  ([ds estimator]
   (predict ds estimator (ds-mod/inference-target-column-names ds))))




(defn transform
  "Calls `transform` on the given sklearn estimator object, and returns the result as a tech.ml.dataset"
  ([ds estimator]
   (transform ds estimator {}))
  ([ds estimator kw-args]
   (let [X (-> ds cf/feature ds->X)
         raw-result (py. estimator transform X)
         X-transformed (raw-tf-result->ds raw-result)
         target-ds (cf/target ds)]
     (if (nil? target-ds)
       X-transformed
       (ds/append-columns
        X-transformed
        (cf/target ds))))))


(defn model-attribute-names [sklearn-model]

  (->> (py/dir sklearn-model)
   (filter #(and  (clojure.string/ends-with? % "_")
               (not (clojure.string/starts-with? % "_"))))))

(defn save-py-get-attr [sklearn-model attr]
  ;; can fail in some cases
  (try (py/get-attr sklearn-model attr)
       (catch Exception e nil)))




(defn model-attributes [sklearn-model]
  (apply merge
         (map
          (fn [attr]
            (hash-map (keyword attr)
                      (py/->jvm (save-py-get-attr sklearn-model attr))))
          (model-attribute-names sklearn-model))))

(comment
  (py/run-simple-string "import numpy as np")
  (def env (py/run-simple-string "x=np.array([1, 2, 3, 4])"))
  (def env (py/run-simple-string "y=np.array([[1], [2]])"))
  (def env (py/run-simple-string "x_shape=x.shape"))
  (def env (py/run-simple-string "y_shape=y.shape"))

  (->
   (py/get-item (:globals env) "x")
   (py/->jvm))
  ;; => [1 2 3 4]

  (->
   (py/get-item (:globals env) "x_shape")
   (py/->jvm))
  ;; => [4]

  (->
   (py/get-item (:globals env) "y")
   (py/->jvm))
  ;; => [[1] [2]]

  (->
   (py/get-item (:globals env) "y_shape")
   (py/->jvm))
  ;; => [2 1]

  (->
   (py.- (py/get-item (:globals env) "x") shape)
   (py/->jvm)))
