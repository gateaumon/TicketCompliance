# Detroit Blight Ticket Compliance Predictor

My implementation for predicting whether a given blight ticket will be paid. Information about the task and data can be found on the [Kaggle competition website](https://www.kaggle.com/c/detroit-blight-ticket-compliance/overview). Classification is done with sklearn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html).

## Use

The script is called with

    ./CompliancePrediction.py path/to/address_file path/to/latlon_file path/to/train_file  path/to/test_file

where:

* address_file is the competition file containing ticket_ids and the corresponding violation address
* latlon_file is the competition file containing violation address and the corresponding latitude and longitude
* train_file is the file containing the training data (features and target)
* test_file is the file containing the test data (features only)

If the PredictionPipeline is imported into your code, you can load a saved model instead of fitting a new one.