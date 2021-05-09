import sys
import shutil
import os

file_count = 0


if sys.argv[3] == '-i':
	for root, dirs, files in os.walk(sys.argv[1]):
		for file in files:
			if sys.argv[4] == '-ext':
				if file.endswith("." + sys.argv[5]):
					file_count = file_count + 1
					shutil.copy(root + '/' + file, sys.argv[2])
					print(file + ' is copying to ' + sys.argv[2])

				if sys.argv[6] == '-c':
					if str(file_count) == sys.argv[7]:
						sys.exit()		
			else:
				file_count = file_count + 1
				shutil.copy(root + '/' + file, sys.argv[2])
				print(file + ' is copying to ' + sys.argv[2])

				if sys.argv[4] == '-c':
					if file_count == sys.argv[5]:
						sys.exit()
else:
	for file in files:
			if sys.argv[3] == '-ext':
				if file.endswith("." + sys.argv[4]):
					file_count = file_count + 1
					shutil.copy(root + '/' + file, sys.argv[2])
					print(file + ' is copying to ' + sys.argv[2])

				if sys.argv[5] == '-c':
					if file_count == sys.argv[6]:
						sys.exit()
			else:
				file_count = file_count + 1
				shutil.copy(root + '/' + file, sys.argv[2])
				print(file + ' is copying to ' + sys.argv[2])

				if sys.argv[3] == '-c':
					if file_count == sys.argv[4]:
						sys.exit()