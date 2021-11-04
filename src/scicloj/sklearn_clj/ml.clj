(ns scicloj.sklearn-clj.ml
  (:require
   [camel-snake-kebab.core :as csk]
   [clojure.string :as str]
   [libpython-clj2.python :refer [->jvm as-jvm cfn path->py-obj py.-]]
   [scicloj.metamorph.ml :as ml]
   [scicloj.sklearn-clj :as sklearn]
   [tech.v3.dataset :as ds]
   [tech.v3.dataset.modelling :as ds-mod]
   [tech.v3.dataset.tensor :as dst]))

(def filter-map
  {"classifier" "classification"
   "regressor"  "regression"})
   
(defn- make-train-fn [module-name estimator-class-name]
  (fn [feature-ds label-ds options]
    (let [dataset (-> (ds/append-columns feature-ds label-ds)
                      (ds-mod/set-inference-target (first (ds/column-names label-ds))))
          estimator (sklearn/fit dataset
                                 (str "sklearn." module-name)
                                 estimator-class-name

                                 (dissoc options :model-type))]
      {:model estimator
       :attributes (sklearn/model-attributes estimator)})))

                   


  


(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps options model-data] :as model}]
  (sklearn/predict feature-ds thawed-model target-columns))

(defn make-names  [f]
  (let [class-name
        (py.- (second  f) __name__)
        module-name
        (->
         (py.- (second  f) __module__)
         (str/replace-first "sklearn." ""))]
    {:module-name module-name :class-name class-name}))


(defn- thaw-fn
  [model-data]
  (:model model-data))

(defn define-estimators! [filter-s]
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
                              "OutputCodeClassifier" "StackingClassifier" "VotingClassifier"}
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
           (make-train-fn module-name class-name)
           predict
           {:thaw-fn tha
            :documentation {:doc-string doc-string}
            :options
            (map (fn [[k v]]
                   {:name (csk/->kebab-case-keyword k)
                    :default v})
                 params)})))

           
     names)))




(define-estimators! "regressor")
(define-estimators! "classifier")

