import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from joblib import dump, load
import time


class PredictionPipeline:
    def __init__(self, address_path=None, latlon_path=None, clf_file=None):
        # load from file
        if clf_file:
            self.my_pipeline = load(clf_file)
        # create a new clf
        else:
            self.clf = GradientBoostingClassifier(random_state=416)
            self.my_pipeline = None
        self.address_path = address_path
        self.latlon_path = latlon_path

    def make_pipeline(self, X_train):
        """
        Creates the pipeline including preprocessing steps and the classifier.
        :param X_train: dataset to determine numerical and categorical columns from
        """
        # handling for numerical columns
        numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]
        numerical_transformer = SimpleImputer()

        # handling for categorical columns
        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # set up column transformer
        column_transformer = ColumnTransformer(transformers=[
            ('numerical', numerical_transformer, numerical_cols),
            ('categorical', categorical_transformer, categorical_cols)
        ])

        # set up preprocessing pipeline
        preprocessor = Pipeline(steps=[
            ('latlon', self.LatLonMerge(self.address_path, self.latlon_path)),
            ('disposition', self.DispositionSimplify()),
            ('column_transform', column_transformer),
        ])

        # set up full pipeline
        self.my_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('gbc', GradientBoostingClassifier(learning_rate=0.1,
                                               n_estimators=150,
                                               max_depth=3,
                                               random_state=416))
        ])

    def fit(self, X_train, y_train):
        """
        Wrapper for Pipeline's fit() function. First sets up the pipeline then fits.
        :param X_train: dataframe of tickets and their features
        :param y_train: class labels for the corresponding tickets in X_train
        """
        self.make_pipeline(X_train)
        print('Fitting...')
        start = time.time()
        self.my_pipeline.fit(X_train, y_train)
        elapsed_time = time.time() - start
        print(f'... Done. Time elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    def predict_proba(self, X_test):
        """
        Wrapper for Pipeline's predict_proba(). Makes predictions based on the learned model,
        saves them to a csv file, and returns them. Columns of the csv file are the ticket_id
        and the corresponding probability that that ticket will be complied with.
        :param X_test: dataframe of tickets and their features
        :return: pd.Series of probabilities with ticket_id as index
        """
        y_scores = self.my_pipeline.predict_proba(X_test)[:, 1]
        y_scores = pd.Series(y_scores, index=X_test.index, name='compliance_proba')
        y_scores.to_csv('predictions.csv')
        return y_scores

    def save_model(self, filename):
        """
        Saves the Pipeline model to a joblib file.
        :param filename: file name for the saved model
        """
        dump(self.my_pipeline, filename)

    class LatLonMerge(BaseEstimator, TransformerMixin):
        """
        Transformer for merging latitude and longitude data into the full datasets. For use
        in Pipeline.
        """
        def __init__(self, address_path, latlon_path):
            self.addresses = pd.read_csv(address_path)
            self.latlons = pd.read_csv(latlon_path)
            self.latlons_with_id = pd.merge(self.addresses, self.latlons, how='left', on='address')
            self.latlons_with_id = self.latlons_with_id.drop(['address'], axis=1)
            self.latlons_with_id = self.latlons_with_id.set_index('ticket_id')

        def fit(self, *_):
            return self

        def transform(self, X, *_):
            return pd.merge(X, self.latlons_with_id, how='left', left_index=True, right_index=True)

    class DispositionSimplify(BaseEstimator, TransformerMixin):
        """
        Transformer for simplifying the disposition column. Combines any label including
        'Fine Waived' into one label. For use in Pipeline.
        """
        def __init__(self):
            pass

        def fit(self, *_):
            return self

        def transform(self, X, *_):
            X['disposition'] = X['disposition'].str.replace('.*Fine Waived.*', 'Fine Waived', regex=True)
            return X
