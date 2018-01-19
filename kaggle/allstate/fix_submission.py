import csv
import re

newrows = []

with open('submission.csv', 'r') as data_file:
	reader = csv.reader(data_file)       
	reader.next()
	for row in reader:
		newrows.append(re.sub('[^0-9,.]','', ','.join(row)))

with open('submission.csv', 'w') as data_file:
	data_file.write('id,loss')
	for row in newrows:
		data_file.write(row + '\n')