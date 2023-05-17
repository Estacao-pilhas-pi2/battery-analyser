import sys

from pprint import pprint

import battery_analyser as ba


for file in sys.stdin:
    prediction = ba.predict(file.strip())
    pprint(prediction)

