## Support Scripts

#### 1. baseline.sh
Run the baseline analysis for given dataset
- Args
  * $1 - Path to positive blow5 file
  * $2 - Path to negative blow5 file
  * $3 - Path to positive fastq file
  * $4 - Path to negative fastq file
  * $5 - Path to positive reference gemome
  * $6 - Path to negative reference gemome
  * $7 - Required base length
```
sh baseline.sh covid.blow5 zymo.blow5 covid.fastq zymo.fastq covid.ref.fasta zymo-ref.fasta 300
```

#### 2. trim-slow5-reads.sh
Provide trimmed Slow5 reads with given raw signal length
- Args
  * $1 - Slow5 file
  * $2 - Expected raw signal length to have after trimming
```
sh trim-slow5-reads.sh covid.slow5 4500
```

#### 3. fast5_filter.py
Filter single_fast5 files in a directory with read length or compared to a list
- Args
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
- Args
  * f5 - Path to fast5 directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python fast5_plotter.py -f5 <fast5_dir> -mad 5
```

#### 5. mycopy.py
Custom copy script with enhanced uses to copy/move/delete/rename
- Args
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
#### 6. npy_plotter.py
Plot the numpy dump files
- Args
  * np - The path to npy files directory
  * mad - Median absolute deviation value for data normalization (default=3)
  * rep   - Is repeatedly normalized or not (default=False)
```
python npy_plotter.py -np <npy_dir> -mad 5
```
#### 7. fast5_reader.py
Print reads in a fast5 file
- Args
  * ft - The path to fast5 files directory
```
python fast5_reader.py -f5 <fast5_dir>
```
#### 8. npy_reader.py
Print data array in numpy dump
- Args
  * np - The path to npy file
```
python npy_reader.py -np <npy_file>
```
