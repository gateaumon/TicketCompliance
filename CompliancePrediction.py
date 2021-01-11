#!/usr/bin/env python

import sys
import json
from Dataset import Dataset
from ClassifierWrapper import ClassifierWrapper


def main():
    try:
        param_file = sys.argv[1]
    except IndexError:
        print('Please provide the location of the json param file.')
        sys.exit(1)

    with open(param_file, 'r') as p:
        params = json.load(p)

    dataset = Dataset(**params['data'])

    clf = ClassifierWrapper(**params['clf'])
    clf.fit(dataset.get_xtrain(), dataset.get_ytrain())
    clf.display_results()
    clf.display_best_params()
    clf.save_model('gbc.joblib')
    clf.predict_proba(dataset.get_xtest())


if __name__ == '__main__':
    main()
