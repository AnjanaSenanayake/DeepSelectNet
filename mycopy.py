import sys
import shutil
import os

file_count = 0


if len(sys.argv)>3 and sys.argv[3] == '-i':
	for root, dirs, files in os.walk(sys.argv[1]):
		for file in files:
			if len(sys.argv)>4 and sys.argv[4] == '-ext':
				if file.endswith("." + sys.argv[5]):
					file_count = file_count + 1
					shutil.copy(root + '/' + file, sys.argv[2])
					print(file + ' is copying to ' + sys.argv[2])

				if len(sys.argv)>6 and sys.argv[6] == '-c':
					if str(file_count) == sys.argv[7]:
						sys.exit()		
			else:
				file_count = file_count + 1
				shutil.copy(root + '/' + file, sys.argv[2])
				print(file + ' is copying to ' + sys.argv[2])

				if len(sys.argv)>4 and sys.argv[4] == '-c':
					if str(file_count) == sys.argv[5]:
						sys.exit()
else:
	for root, dirs, files in os.walk(sys.argv[1]):
		for file in files:
				if len(sys.argv)>3 and sys.argv[3] == '-ext':
					if file.endswith("." + sys.argv[4]):
						file_count = file_count + 1
						shutil.copy(root + '/' + file, sys.argv[2])
						print(file + ' is copying to ' + sys.argv[2])

					if len(sys.argv)>5 and sys.argv[5] == '-c':
						if str(file_count) == sys.argv[6]:
							sys.exit()
				else:
					file_count = file_count + 1
					shutil.copy(root + '/' + file, sys.argv[2])
					print(file + ' is copying to ' + sys.argv[2])

					if len(sys.argv)>3 and sys.argv[3] == '-c':
						if str(file_count) == sys.argv[4]:
							sys.exit()
