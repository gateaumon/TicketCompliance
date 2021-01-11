from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from joblib import dump, load
import time


class ClassifierWrapper:
    def __init__(self, clf_file=None, param_grid=None, scoring='roc_auc'):
        # load from file
        if clf_file:
            self.grid = load(clf_file)
        # create a new clf
        else:
            clf = GradientBoostingClassifier(random_state=416)
            self.grid = GridSearchCV(clf, param_grid, scoring=scoring)

    def fit(self, X_train, y_train):
        """
        Wrapper for GridSearchCV's fit() function. Trains the GradientBoostingClassifier
        based on the best combination of parameters found in the grid search.
        :param X_train: dataframe of tickets and their features
        :param y_train: class labels for the corresponding tickets in X_train
        """
        print('Fitting...')
        start = time.time()
        self.grid.fit(X_train, y_train)
        elapsed_time = time.time() - start
        print(f'... Done. Time elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')

    def predict_proba(self, X_test):
        """
        Wrapper for GridSearchCV's predict_proba(). Makes predictions based on the learned model
        and saves them to a csv file. Columns are the ticket_id and the corresponding probability
        that that ticket will be complied with.
        :param X_test: dataframe of tickets and their features
        """
        y_scores = self.grid.predict_proba(X_test)[:, 1]
        pd.Series(y_scores, index=X_test.index, name='compliance_proba').to_csv('predictions.csv')

    def save_model(self, filename):
        """
        Saves the GridSearchCV model to a joblib file.
        :param filename: file name for the saved model
        """
        dump(self.grid, filename)

    def display_results(self):
        """
        Displays the mean score and standard deviation for each combination of features used in
        the grid search.
        """
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        result_cols = [key for key in self.grid.cv_results_.keys() if key.startswith('param_')]
        result_dict = {key: self.grid.cv_results_[key] for key in result_cols}
        df = pd.DataFrame(result_dict)
        df['Mean Score'] = self.grid.cv_results_['mean_test_score']
        df['Standard Deviation'] = self.grid.cv_results_['std_test_score']

        print('\nGrid scores on development set:')
        print(df)

    def display_best_params(self):
        """
        Displays the best combination of parameters found in the grid search.
        """
        print('\nBest results found on development set:')
        print(pd.Series(self.grid.best_params_).to_string())
