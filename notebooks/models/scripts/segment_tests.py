import unittest
import pandas as pd
import helper_functions as hf


class TestDataSegment(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(start='2018-01-01', periods=10, freq='d')
        data = range(0, 10)
        self.time_column = 'datetime'
        self.df = pd.DataFrame(data={self.time_column: dates, 'data': data})
        index = self.time_column + 'Index'
        self.df[index] = self.df[self.time_column]
        self.df = self.df.set_index(index)

    def test_segment_even_windows(self):
        segments, targets = hf.segment_dataset(self.df, self.time_column,
                                               x_win_size=pd.Timedelta(1, unit='d'),
                                               y_win_size=pd.Timedelta(1, unit='d'),
                                               shift=pd.Timedelta(1, unit='d'))
        print(segments)
        print(targets)
        self.assertEqual(len(segments), len(targets))
        self.assertEqual(len(segments), 8)

    def test_segment_odd_windows(self):
        segments, targets = hf.segment_dataset(self.df, self.time_column,
                                               x_win_size=pd.Timedelta(2, unit='d'),
                                               y_win_size=pd.Timedelta(1, unit='d'),
                                               shift=pd.Timedelta(1, unit='d'))
        print(segments)
        print(targets)
        self.assertEqual(len(segments), len(targets))
        self.assertEqual(len(segments), 7)

    def test_segment_too_big_windows(self):
        segments, targets = hf.segment_dataset(self.df, self.time_column,
                                               x_win_size=pd.Timedelta(8, unit='d'),
                                               y_win_size=pd.Timedelta(8, unit='d'),
                                               shift=pd.Timedelta(1, unit='d'))
        print(segments)
        print(targets)
        self.assertEqual(len(segments), len(targets))
        self.assertEqual(len(segments), 0)

    def test_extract_percentile(self):
        segments, targets = hf.segment_dataset(self.df, self.time_column,
                                               x_win_size=pd.Timedelta(2, unit='d'),
                                               y_win_size=pd.Timedelta(2, unit='d'),
                                               shift=pd.Timedelta(1, unit='d'))
        print(segments)
        print(targets)
        x_df = hf.extract_percentile(segments, self.time_column, 0.5)
        y_df = hf.extract_percentile(targets, self.time_column, 0.5)

        # Check that the number of rows matches the number of windows in segments/targets
        self.assertEqual(x_df.shape[0], len(segments))
        self.assertEqual(y_df.shape[0], len(targets))

        # Calculate average values from segments/targets
        x_averages = []
        for window in segments:
            x_averages.append(sum(window['data']) / len(window['data']))
        y_averages = []
        for window in targets:
            y_averages.append(sum(window['data']) / len(window['data']))

        # Check that our calculated averages match each row in the extracted dataframes
        for i in range(0, len(x_averages)):
            self.assertAlmostEqual(x_averages[i], x_df['data'].values[i])
        for i in range(0, len(y_averages)):
            self.assertAlmostEqual(y_averages[i], y_df['data'].values[i])

    # TODO Add more tests for extract


if __name__ == '__main__':
    unittest.main()
