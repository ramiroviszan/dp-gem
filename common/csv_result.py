import os
import csv
from datetime import datetime

class CSVResult():

    def __init__(self, results_fullpath, header):
        self.results_fullpath = results_fullpath
        if not os.path.isfile(results_fullpath):
            with open(results_fullpath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['date', *header])
    
    def save_results(self, values):
        with open(self.results_fullpath, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.timestamp(datetime.now()), *values])

