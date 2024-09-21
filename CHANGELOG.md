# Change Log
All notable changes to this project will be documented in this file. This change log follows the conventions of [keepachangelog.com](http://keepachangelog.com/).

* unreleased
  * support single-case-capital params (#5)
  * fixes #8 regressors use predict_prob by default, which is not defined or most/all regressors
  * fixes #9 certain options cannot be passed to models
  * fixes #10 FixedThresholdClassifier and TunedThresholdClassifierCV 

* 0.4.0
  * fix result of predict to be a probability distribution
  * fixed serialization of contexts containing sklearn-clj models
  * allow reverse-mapping of categorical variables of prediction
* 0.3.6
  * added model attributes to train results
* 0.3.5
  * require full module path for estimators
  * added method to retrieve all attributes of trained model as map
  
* 0.3.0
  * adapted to libpython-clj 2.0.0-BETA-12
  * zero copy
* 0.2.1 
  * added fit-transform
  * more robust
  * more tests 
* 0.2.0 
  * adapted to latest metamorph
* 0.1.0 
  * first version
