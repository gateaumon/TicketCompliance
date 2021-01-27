# Detroit Blight Ticket Compliance Predictor

My implementation for predicting whether a given blight ticket will be paid. Information about the task and data can be found on the [Kaggle competition website](https://www.kaggle.com/c/detroit-blight-ticket-compliance/overview). Classification is done with sklearn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

## Use

The script is called with

    ./CompliancePrediction.py path/to/params.json

where params.json contains the desired parameters for feature selection and classifier creation. Parameters are split into two sections: "data" and "clf". Here is an example parameter file:

```json
{
	"data": {"train_file": "data/train.csv", 
		 "test_file": "data/test.csv", 
		 "feature_names": ["fine_amount", "late_fee", "discount_amount", "judgment_amount", "disposition"],
		 "locations": {"address_file": "data/addresses.csv", 
		 	       "latlon_file": "data/latlons.csv"},
		 "dispositions": true},
	"clf": {"param_grid": {"learning_rate": [0.2, 0.1, 0.01], 
			       "n_estimators": [50, 100, 150], 
			       "max_depth": [2, 3, 4]},
		"scoring": "roc_auc"}
}
```

### Data Parameters

The data parameters should be under the JSON key "data" and are passed during initialization of the Dataset class. Required parameters for "data" are:

```json
{
	"train_file": "path/to/train_data",
	"test_file": "path/to/test_data",
	"feature_names": ["columns", "to", "use", "as", "features"]
}
```

If you want to add features for latitude and longitude of the violation location, you must include a "locations" key whose value is another JSON object containing the following key-value pairs:

```json
{
    "address_file": "path/to/address_file",
    "latlon_file": "path/to/latlon_file"
}
```

If you want to add one-hot features for disposition labels, include the following:

    "dispositions": true

### Classifier Parameters

Classifier parameters should be under the JSON key "clf" and are passed during initialization of the ClassifierWrapper class. If you are loading a previously saved GridSearchCV model, the following parameter is required:

	"clf_file": "path/to/clf_file"


Otherwise, the "param_grid" parameter is required, which corresponds to the parameter of the same name used by GridSearchCV.

The default scoring metric used is Area Under the Receiver Operating Characteristic curve. If the user wants to use a different scoring metric, this can be specified with:

    "scoring": "other_metric"
