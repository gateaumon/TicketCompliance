import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class Dataset:
    def __init__(self, train_file, test_file, feature_names, **kwargs):
        self.train_data = None
        self.test_data = None
        self.feature_names = feature_names
        self.read_data(train_file, feature_names, is_train=True)
        self.read_data(test_file, feature_names)
        if 'locations' in kwargs:
            self.add_location_features(**kwargs['locations'])
        if 'dispositions' in kwargs:
            self.add_disposition_features()

    def read_data(self, filename, column_names, is_train=False):
        """
        Reads data from csv files.
        :param filename: location of the csv file
        :param column_names: columns to read in
        :param is_train: whether the file contains training data (which means the 'compliance'
                column containing y-values should also be read in
        """
        if is_train:
            df = pd.read_csv(filename, usecols=column_names + ['compliance', 'ticket_id'], encoding='cp1252')
            self.train_data = df.dropna().reset_index(drop=True)
        else:
            self.test_data = pd.read_csv(filename, usecols=column_names + ['ticket_id'])

    def add_location_features(self, address_file, latlon_file):
        """
        Adds features for latitude and longitude of the blight violation location to the train
        and test data. Fills missing values in the test data with the average latitude and longitude
        of the train data.
        :param address_file: csv file mapping ticket_id to addresses
        :param latlon_file: csv file mapping addresses to latitudes and longitudes
        """
        # first get addresses
        self.merge_addresses(address_file)

        # now get latitudes and longitudes
        self.merge_latlons(latlon_file)

        # drop NaN rows in train_data
        self.train_data = self.train_data.dropna().reset_index(drop=True)

        # set the missing lats and lons in test_data to the means of train_data
        mean_lat = self.train_data['lat'].mean()
        mean_lon = self.train_data['lon'].mean()
        self.test_data['lat'] = self.test_data['lat'].fillna(mean_lat)
        self.test_data['lon'] = self.test_data['lon'].fillna(mean_lon)

        # add new location features to feature_names
        self.feature_names.extend(['lat', 'lon'])

    def merge_addresses(self, address_file):
        """
        Merges the addresses file with the train and test dataframes.
        :param address_file: file containing ticket_id-address mappings
        """
        addresses = pd.read_csv(address_file)
        self.train_data = pd.merge(self.train_data, addresses, how='left', on='ticket_id')
        self.test_data = pd.merge(self.test_data, addresses, how='left', on='ticket_id')

    def merge_latlons(self, latlon_file):
        """
        Merges the latlons file with the train and test dataframes.
        :param latlon_file: file containing address-lat/lon mappings
        """
        latlons = pd.read_csv(latlon_file)
        self.train_data = pd.merge(self.train_data, latlons, how='left', on='address')
        self.test_data = pd.merge(self.test_data, latlons, how='left', on='address')

    def add_disposition_features(self):
        """
        Add one-hot features for the disposition column. This column contains the court ruling
        on the ticket (responsible, not responsible, etc.)
        """
        # combine all fine-waived dispositions into one category
        self.train_data['disposition'] = self.train_data['disposition'].str.replace('.*Fine Waived.*',
                                                                                    'Fine Waived',
                                                                                    regex=True)
        self.test_data['disposition'] = self.test_data['disposition'].str.replace('.*Fine Waived.*',
                                                                                  'Fine Waived',
                                                                                  regex=True)

        # convert disposition categories into numerical labels
        # use a dict of the categories to handle unseen values in test
        le = LabelEncoder().fit(self.train_data['disposition'])
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        # use 999 as encoding for unknown labels
        self.train_data['disp_labelenc'] = self.train_data['disposition'].apply(lambda x: le_dict.get(x, 999))
        self.test_data['disp_labelenc'] = self.test_data['disposition'].apply(lambda x: le_dict.get(x, 999))

        # convert dispositions into one-hot columns
        onehot = OneHotEncoder(handle_unknown='ignore', sparse=False)
        onehot.fit(self.train_data[['disp_labelenc']])
        train_disp_enc = pd.DataFrame(onehot.transform(self.train_data[['disp_labelenc']]), columns=le.classes_)
        test_disp_enc = pd.DataFrame(onehot.transform(self.test_data[['disp_labelenc']]), columns=le.classes_)

        # add one-hot columns to original train and test sets
        self.train_data = pd.concat([self.train_data, train_disp_enc], axis=1)
        self.test_data = pd.concat([self.test_data, test_disp_enc], axis=1)

        # add new disposition features to feature_names and remove original disposition column
        self.feature_names.extend(le.classes_.tolist())
        self.feature_names.remove('disposition')

    def get_xtrain(self):
        """
        Returns the feature columns of the training data. For use in training the classifier.
        """
        return self.train_data.loc[:, self.feature_names]

    def get_ytrain(self):
        """
        Returns the class labels of the training data. For use in training the classifier.
        """
        return self.train_data.loc[:, 'compliance']

    def get_xtest(self):
        """
        Returns the feature columns of the test data with the ticket_id as the index.
        """
        tst = self.test_data.loc[:, self.feature_names + ['ticket_id']]
        return tst.set_index('ticket_id')
