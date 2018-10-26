from writer import writefile

import json


def test_writer():
    dict = {}
    dict["h"] = 6
    dict["cat"] = 'dog'
    dict["favorite number"] = 22

    file = writefile(dict, 'testingjson')

    with open('testingjson.json', 'r') as prac:
        readdict = json.load(prac)

    assert readdict == dict
