import os
import pandas as pd
from collections import defaultdict

class Logger:
    def __init__(self, path, name):
        self.data = defaultdict(list)
        self.dir_path = os.path.join(os.getcwd(), path)
        self.file_path = os.path.join(self.dir_path, name)

    def save(self):
        try:
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path)
            df = pd.DataFrame(self.data)
            df.to_csv(f'{self.file_path}.csv', index=False, encoding='utf-8')

        except OSError:
            print("Error: Failed to create the directory.")

    def append(self, dic):
        for k, v in dic.items():
            self.data[k].append(v)

