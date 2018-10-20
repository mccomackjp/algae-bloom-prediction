#!/usr/bin/python

import csv, json
import sys

csvFilePath = ""

entries = []

if __name__ == "__main__":
	if (len(sys.argv) < 2):
		print("Invalid syntax.\n  Rerun the script with <CSV_File_Path> <JSON_File_name>")
		print("  If no JSON file name is supplied the default 'temp.json' will given.")
		quit()
	else:
		csvFilePath = sys.argv[1]
		if (len(sys.argv) == 3):
			jsonFileName = sys.argv[2]
		else:
			jsonFileName = "temp.json"
		with open(jsonFileName, "w") as jsonFile:
			with open(csvFilePath) as csvFile:
				csvReader = csv.DictReader(csvFile)
				for row in csvReader:
					entries.append(row)
			jsonFile.write(json.dumps(entries, indent=4))
