import pandas as pd


class TicketDataReader:
    def __init__(self):
        self.columns = ['ticket_id',
                        'fine_amount',
                        'late_fee',
                        'discount_amount',
                        'judgment_amount',
                        'disposition']

    def read_train_data(self, filepath):
        """
        Reads in the desired columns of the training data. Index is set to ticket_id.
        Feature columns and target column returned separately.
        :param filepath: location of training data
        :return: pd.DataFrame containing feature columns and pd.Series containing the
                 target column
        """
        df = pd.read_csv(filepath, usecols=self.columns + ['compliance'], encoding='cp1252')
        df = df[df['compliance'].notnull()]
        df = df.set_index('ticket_id')
        y = df.compliance
        x = df.drop(['compliance'], axis=1)

        return x, y

    def read_test_data(self, filepath):
        """
        Reads in the desired columns of the test data. Index is set to ticket_id.
        :param filepath: location of the test data
        :return: pd.DataFrame containing feature columns
        """
        df = pd.read_csv(filepath, usecols=self.columns)
        df = df.set_index('ticket_id')

        return df
