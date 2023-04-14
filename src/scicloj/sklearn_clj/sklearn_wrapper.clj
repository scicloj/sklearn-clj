(ns scicloj.sklearn-clj.sklearn-wrapper
  (:require
   [libpython-clj2.python :as py]
   [scicloj.metamorph.core :as mm]
   [scicloj.sklearn-clj]
   [tablecloth.api :as tc]
   [tech.v3.dataset :as ds]))


(py/initialize!)

(py/from-import sklearn.base BaseEstimator)
(py/from-import sklearn.base ClassifierMixin)


(defn set-column-names [ds column-name-list]
  (-> ds
      (tc/rename-columns (zipmap (tc/column-names ds)
                                 column-name-list))))


(defn pipe-fn->estimator [class-name pipe-fn X-column-names y-column-names X-categorical-maps y-categorical-maps]
  (let [fitted-state (atom {})
        estimator-class (py/create-class
                         class-name
                         [BaseEstimator,ClassifierMixin]
                         {

                          "predict"
                          (py/make-instance-fn
                           (fn [this X]
                             (let [X-ds (scicloj.sklearn-clj/ndarray->ds X)
                                   ds (-> X-ds
                                          (set-column-names X-column-names)
                                          (tc/add-column :species nil)

                                          (ds/assoc-metadata y-column-names :categorical-map y-categorical-maps))
                                   prediction (->
                                               (mm/transform-pipe ds pipe-fn @fitted-state)
                                               :metamorph/data
                                               :species
                                               vec)]
                               (println :prediction prediction)
                               prediction)))

                          "fit"
                          (py/make-instance-fn
                           (fn [self X y]
                             ;; (println :fit-X-shape (py/py.- X shape))
                             ;; (println :fit-y-shape (py/py.- y shape))
                             (let [X-ds (scicloj.sklearn-clj/ndarray->ds X)
                                   y-ds (scicloj.sklearn-clj/ndarray->ds y)
                                   labels
                                   (map long
                                        (-> y-ds (get 0) (distinct)))
                                   ds (->
                                       (tc/add-column X-ds
                                                      (tc/column-count X-ds)
                                                      (get  y-ds 0))
                                       (set-column-names (concat X-column-names y-column-names))
                                       (ds/assoc-metadata y-column-names
                                                          :categorical-map
                                                          (update-in y-categorical-maps [:lookup-table] #(select-keys % labels))))]



                               (reset! fitted-state (mm/fit-pipe ds pipe-fn)))


                             self))})]
    (estimator-class)))
