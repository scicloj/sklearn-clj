(ns scicloj.perf-test
  (:require  [clojure.test :as t]
             [tablecloth.api :as tc]
             [tech.v3.dataset.tensor :as dst]
             [libpython-clj2.python.np-array]
             [scicloj.sklearn-clj :refer [fit]]
             [libpython-clj2.python  :as py]))

(defmacro labeled-time
  "Evaluates expr and prints the time it took.  Returns the value of
 expr."
  {:added "1.0"}
  [label expr]
  `(let [start# (. System (nanoTime))
         ret# ~expr]
     (prn (str label  " - " "elapsed time: " (/ (double (- (. System (nanoTime)) start#)) 1000000.0) " msecs"))
     ret#))

(t/deftest train-flight-python []
  (let  [res
         (py/run-simple-string (slurp "train_delay.py"))

         python-time
         (-> res :globals (get "execution_time"))


         train
         (-> res :globals
             (get "tensor_train")
             py/->jvm
             dst/tensor->dataset)


         y
         (-> res :globals
             (get "y_train")
             py/->jvm)

         train
         (-> train
             (tc/add-column :y y)
             (tech.v3.dataset.modelling/set-inference-target :y))
         start (System/currentTimeMillis)
         _ (fit train :sklearn.ensemble :GradientBoostingClassifier)
         end (System/currentTimeMillis)
         clojure-time (/ (- end start) 1000)]
    (t/is (< python-time 20))
    (t/is (< clojure-time 20))))
