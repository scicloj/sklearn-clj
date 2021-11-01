(ns scicloj.sklearn-clj.ml
  (:require [scicloj.metamorph.ml :as ml]
            [clojure.string :as str]
            [scicloj.sklearn-clj :as sklearn]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.tensor :as dst]
            [tech.v3.dataset.modelling :as ds-mod]
            [libpython-clj2.python :refer [->jvm as-jvm call-attr call-attr-kw cfn py.- py. py.. python-type path->py-obj as-python]]
            [camel-snake-kebab.core :as csk]))
            
  


(def filter-map
  {"classifier" "classification"
   "regressor"  "regression"})
   
(defn- make-train-fn [module-name estimator-class-name]
  (fn [feature-ds label-ds options]
    (let [dataset (-> (ds/append-columns feature-ds label-ds)
                      (ds-mod/set-inference-target (first (ds/column-names label-ds))))]
                      
      (sklearn/fit dataset (str "sklearn."  module-name) estimator-class-name
                   ;; options
                   (dissoc options :model-type)))))
                   


  


(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps options model-data] :as model}]
  (sklearn/predict feature-ds model-data target-columns))

(defn make-names  [f]
  (let [class-name
        (py.- (second  f) __name__)
        module-name
        (->
         (py.- (second  f) __module__)
         (str/replace-first "sklearn." ""))]
    {:module-name module-name :class-name class-name}))




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
           {:documentation {:doc-string doc-string}
            :options
            (map (fn [[k v]]
                   {:name (csk/->kebab-case-keyword k)
                    :default v})
                 params)})))

           
     names)))




(define-estimators! "regressor")
(define-estimators! "classifier")

(comment
  (def ds (-> (dst/tensor->dataset [[0, 0 0 ], [1, 1 1 ], [2, 2 2]])
              (ds-mod/set-inference-target 2)))
              
  (def trained-model
    (ml/train ds {:model-type :sklearn.classification/ada-boost-classifier}))


  (ml/predict ds trained-model))




(comment)
