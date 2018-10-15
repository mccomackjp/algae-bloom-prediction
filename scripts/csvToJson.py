#!/usr/bin/python

import csv, json
import sys

csvFilePath = ""




if __name__ == "__main__":
	if (len(sys.argv) < 2):
		print("Invalid syntax.\n  Rerun the script with <CSV_File_Path> <JSON_File_name>")
		quit()
	else:
		data = {}
		csvFilePath = sys.argv[1]
		with open(csvFilePath) as csvFile:
			csvReader = csv.DictReader(csvFile)
			for row in csvReader:
				print(row)
				date = row["Date (mm.dd.yyyy)"]
				data[date] = row

		if (len(sys.argv) == 3):
			jsonFileName = sys.argv[2]
		else:
			jsonFileName = "test.json"
		with open(jsonFileName, "w") as jsonFile:
			jsonFile.write(json.dumps(data, indent=4))
	
		