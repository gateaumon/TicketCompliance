#!/usr/bin/env python

import sys
from TicketDataReader import TicketDataReader
from PredictionPipeline import PredictionPipeline


def main():
    try:
        address_path = sys.argv[1]
        latlon_path = sys.argv[2]
        train_path = sys.argv[3]
        test_path = sys.argv[4]
    except IndexError:
        print('Please provide paths to the address file, latlon file, train file, and test file.')
        sys.exit(1)

    data_reader = TicketDataReader()
    X_train, y_train = data_reader.read_train_data(train_path)
    X_test = data_reader.read_test_data(test_path)

    clf = PredictionPipeline(address_path, latlon_path)
    clf.fit(X_train, y_train)
    clf.save_model('gbc.joblib')
    clf.predict_proba(X_test)


if __name__ == '__main__':
    main()
