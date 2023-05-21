import sys

from pprint import pprint

import battery_analyser.predict as ba_predict


for file in sys.stdin:
    prediction = ba_predict.predict(file.strip())
    pprint(prediction)

