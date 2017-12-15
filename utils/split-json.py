'''
Split apart test.json so I can run the damn thing on my computer.
Because apparently I don't have enough memory.
'''

import ijson.backends.yajl2 as ijson
import simplejson as json

jsonFile = open("data/test.json", "rb")

curSplit = 0
objBuffer = []

objs = ijson.items(jsonFile, "item")
for obj in objs:
    objBuffer.append(obj)
    if len(objBuffer) == 1000:
        with open("data/test." + str(curSplit) + ".json", "w") as outfile:
            json.dump(objBuffer, outfile, separators=(',', ':'))
        objBuffer = []
        curSplit += 1
        print("Wrote " + str(curSplit * 1000) + " records to disk")
# dump whatever is left
with open("data/test." + str(curSplit) + ".json", "w") as outfile:
    json.dump(objBuffer, outfile, separators=(',', ':'))
