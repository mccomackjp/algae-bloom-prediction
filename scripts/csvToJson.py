#!/usr/bin/python

import csv, json
import sys

csvFilePath = ""

entries = []

if __name__ == "__main__":
	if (len(sys.argv) < 3):
		print("Invalid syntax.\n  Rerun the script with <CSV_File_Path> <Lake_Name> <Site_Name> <JSON_File_name>")
		print("  If no JSON file name is supplied the default 'temp.json' will given.")
		quit()
	else:
		csvFilePath = sys.argv[1]
		lakeName = sys.argv[2]
		siteName = sys.argv[3]
		if (len(sys.argv) == 5):
			jsonFileName = sys.argv[4]
		else:
			jsonFileName = "temp.json"
		with open(jsonFileName, "w") as jsonFile:
			with open(csvFilePath) as csvFile:
				csvReader = csv.DictReader(csvFile)
				for row in csvReader:
					entries.append(row)
					
			data = {lakeName +"_" + siteName: entries}
			jsonFile.write(json.dumps(data, indent=4))
