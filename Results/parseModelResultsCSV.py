import csv
import json

"""
Parse CSV data into JSON format.
"""

# Load CSV
with open("model_results.csv", newline="") as f:
    reader = csv.reader(f)
    rows = list(reader)

rows = [[x for x in row if x != ""] for row in rows]

for row in rows:
    print(row)

results = {}

j = 0
for i in range(3):
    sheet = []
    while rows[j] != []:
        sheet.append(rows[j])
        j += 1

    headers = sheet[0]

    data = []
    for row in sheet[1:]:
        item = {}
        for h, v in zip(headers, row):
            # convert numeric values to float if possible
            try:
                v = float(v)
            except ValueError:
                pass
            item[h] = v
        data.append(item)
    results[headers[0]] = data

    while j < len(rows) and rows[j] == []:
        j += 1

header = rows[j][0]
j += 1

temp1 = {}
byHead = rows[j][0]
j += 1
while j < len(rows) and rows[j] != []:
    split = rows[j][0].split(":")
    temp1[split[0].strip()] = split[1].strip() + "%"
    j += 1

j += 1
temp2 = {}
byHead = rows[j][0]
j += 1
while j < len(rows) and rows[j] != []:
    split = rows[j][0].split("-")
    temp2[split[0].strip() + " " + split[1].strip()] = split[2].strip()
    j += 1

results[header] = [temp1, temp2]


with open("model_results.json", "w") as f:
    json.dump(results, f, indent=4)
