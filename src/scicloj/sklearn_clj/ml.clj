(ns scicloj.sklearn-clj.ml
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.string :as str]
   [libpython-clj2.python :refer [->jvm as-jvm cfn path->py-obj py.- py.] :as py]
   [scicloj.metamorph.ml :as ml]
   [scicloj.sklearn-clj :as sklearn]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]))

(py/initialize!)
(def pickle (py/import-module "pickle"))
(def filter-map
  {"classifier" "classification"
   "regressor"  "regression"})
   
(defn- make-train-fn [module-name estimator-class-name predict-proba?]
  (fn [feature-ds label-ds options]
    (let [target-column (first (ds/column-names label-ds))
          dataset (-> (ds/append-columns feature-ds label-ds)
                      (ds-mod/set-inference-target target-column))
          model-options (dissoc options :model-type :predict-proba?)
          estimator (sklearn/fit dataset
                                 (str "sklearn." module-name)
                                 estimator-class-name
                                 model-options)]

      {:model estimator
       :predict-proba? (get options :predict-proba? predict-proba?)
       
       :pickled-model (-> (py. pickle dumps estimator)
                          py/->jvm
                          short-array)
       :attributes (sklearn/model-attributes estimator)})))



(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps predict-proba?] :as model}]
  (let [prediction
        (if (-> model :model-data :predict-proba?)
          (sklearn/predict-proba feature-ds thawed-model target-columns)
          (sklearn/predict feature-ds thawed-model target-columns))]

    (-> prediction
        (ds/assoc-metadata target-columns :categorical-map (get target-categorical-maps (first target-columns))))))




(defn make-names  [f]
  (let [class-name
        (py.- (second  f) __name__)
        module-name
        (->
         (py.- (second  f) __module__)
         (str/replace-first "sklearn." ""))]
    {:module-name module-name :class-name class-name}))


(def builtins (py/import-module "builtins"))

(defn- thaw-fn
  [model-data]
  (py/py. pickle loads
        (py/py. builtins bytes (:pickled-model model-data))))


(defn define-estimators! [filter-s predict-proba]
  (let [ estimators
        (->
         (cfn
          (path->py-obj "sklearn.utils.all_estimators")
          :type_filter filter-s) as-jvm)
        names
        (->> (map make-names estimators)
             (filter #(not (contains?
                            #{"MultiOutputRegressor" "RegressorChain" "StackingRegressor" "VotingRegressor"
                              "ClassifierChain" "MultiOutputClassifier" "OneVsOneClassifier" "OneVsRestClassifier"
                              "OutputCodeClassifier" "StackingClassifier" "VotingClassifier" "FixedThresholdClassifier"
                              "TunedThresholdClassifierCV"}
                            (:class-name %)))))]

    (run!
     (fn [{:keys [module-name class-name]}]
       ;; (println module-name class-name)
       (let [estimator (sklearn/make-estimator (str "sklearn." module-name) class-name {})
             params
             (->jvm (py. estimator get_params))
             doc-string (py.- estimator __doc__)]
         (ml/define-model!
           (keyword  (str "sklearn." (filter-map filter-s)) (csk/->kebab-case-string class-name))
           (make-train-fn module-name class-name predict-proba)
           predict
           {:thaw-fn thaw-fn
            :documentation {:doc-string doc-string}
            :options
            (vec
             (concat
              (mapv (fn [[k v]]

                      (vector
                       (csk/->kebab-case-keyword k)
                       {:optional true}
                       :any))
                    params)
              [[:predict-proba? {:optional true} boolean?]]))})))

     names)))

(define-estimators! "regressor" false)
(define-estimators! "classifier" true)

