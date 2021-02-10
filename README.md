# sklearn-clj

This library gives easy access to all estimators frm sklearn in a Clojure friendly way.

```clojure
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
    (fit train-ds :linear_model :linear-regression {}))

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

```

## Usage

## License

Copyright © 2021 Carsten Behring

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
# sklearn-clj
