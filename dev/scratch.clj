(ns scicloj.scratch)

(comment

  (require '[scicloj.sklearn-clj.ml]) ;; registers all models
  (require '[scicloj.ml.core :as ml]
           '[scicloj.ml.metamorph :as mm]
           '[scicloj.ml.dataset :as ds]
           '[tech.v3.libs.smile.data :as smile-data]
           '[camel-snake-kebab.core :as csk]
           '[libpython-clj2.python :refer [py. py.-]]
           '[scicloj.sklearn-clj.metamorph :as sklearn-mm]
           '[smile.manifold :as smile-mf])

  (def pinguins
    (->
     (ds/dataset
      "https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv"
      {:key-fn csk/->kebab-case-keyword})))

  (defn fit-smile-umap []
    (fn [{:metamorph/keys [id data mode] :as ctx}]
      (def data data)
      (assoc ctx :metamorph/data
             (-> data
                 (ds/rows :as-double-arrays)
                 smile-mf/umap
                 (#(.coordinates %))
                 ds/dataset))))


  (def pipe
    (ml/pipeline
     (mm/drop-missing)
     (mm/select-columns [:culmen-length-mm :culmen-depth-mm :flipper-length-mm :body-mass-g])
     (mm/std-scale :type/numerical {})
     {:metamorph/id :model}
     (fit-smile-umap)))
  ;; (sklearn-mm/fit-transform :umap "UMAP")



  (def fit-ctx
    (ml/fit-pipe pinguins pipe))

  (-> fit-ctx :metamorph/data))
