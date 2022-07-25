## Support Scripts

#### 1. fastq_trimmer.py
Trim fastq reads to shorten the read to a specific length
- Args
  * fq - Path to fastq file
```
python fastq_trimmer.py -fq <fastq_file>
```
#### 2. baseline.sh
Run the baseline analysis for given dataset
- Args
  * $1 - Path to positive slow5 file
  * $2 - Path to negative slow5 file
  * $3 - Path to positive fastq file
  * $4 - Path to negative fastq file
  * $5 - Path to mixed reference genome (This is the concatenated reference genome of the 2 species)
```
sh baseline.sh <$1> <$2> <$3> <$4> <$5>
```
#### 3. fast5_filter.py
Filter single_fast5 files in a directory with read length or compared to a list
- Options
  * f5 - Path to fast5 directory
  * c - The read signal cutoff value (default=1500)
  * sz - Read signal sample size (default=3000)
  * type - Filter type | Length, List (default=Length)
  * list - List of fast5 names,if the filter type is list
```
python fast5_filter.py -f5 <fast5_dir> -c 1500 -sz 3000 -type Length
```
```
python fast5_filter.py -f5 <fast5_dir> -c 1500 -sz 3000 -type List -list <fast5_list.txt>
```
#### 4. fast5_plotter.py
Plots the raw signal in fast5 files
- Options
  * f5 - Path to fast5 directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python fast5_plotter.py -f5 <fast5_dir> -mad 5
```

#### 5. line_diff.py
Compare 2 files line by line
- Options
  * f1 - The original file
  * f2 - The other file to compare
```
python line_diff.py -f1 <file_1.txt> -f2 <file_2.txt>
```
#### 6. mycopy.py
Custom copy script with enhanced uses to copy/move/delete/rename
- Options
  * s - The path to source of files
  * d - The path to destination to be copied
  * i - Iterate through files including child directories (default=True)
  * ext - File extension type to be copied (default='' -> any)
  * c - Amount of files to be copied/moved/deleted (default=0 -> means all files)
  * cut - Is cut and paste files (default=False)
  * del - Is delete source files (default=False)
  * rn - Suffix to rename files while copying files (default='')
```
python mycopy.py -s <src_dir> -d <dest_dir> -ext fast5 -c 10000 -cut True -rn _zymo
```
#### 7. npy_plotter.py
Plot the numpy dump files
- Options
  * np - The path to npy files directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python npy_plotter.py -np <npy_dir> -mad 5
```
#### 8. fast5_reader.py
Print reads in a fast5 file
- Options
  * ft - The path to fast5 files directory
```
python fast5_reader.py -f5 <fast5_dir>
```
#### 9. npy_reader.py
Print data array in numpy dump
- Options
  * np - The path to npy file
```
python npy_reader.py -np <npy_file>
```
