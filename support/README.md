## Support Scripts

#### 1. dataset_partitioner.sh
Partition given blow5 in to train and test splits with given number of reads for each
- Args
  * $1 - path to source BLOW5
  * $2 - Required minimum read length for train/test sets
  * $3 - Number of reads for train set
  * $4 - Number of reads for test set
  * $5 - Output file/dataset name
```
sh dataset_partitioner.sh covid.blow5 4500 20000 20000 covid
```

#### 2. baseline.sh
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

#### 3. trim-slow5-reads.sh
Provide trimmed Slow5 reads with given raw signal length
- Args
  * $1 - Slow5 file
  * $2 - Expected raw signal length to have after trimming
```
sh trim-slow5-reads.sh covid.slow5 4500
```

#### 4. guppy_fast_minimap2.sh
Evaluator for Guppy_fast+Minimap2 read classification accuracy
- Args
  * $1 - Path to targeted species test data fastq
  * $2 - Path to non-targeted species test data fastq
  * $3 - Path to targeted species reference genome
  * $4 - Path to non-targeted species reference genome
```
python guppy_fast_minimap2.sh test-covid.fastq test-zymo.fastq covid-ref.fasta zymo-ref.fasta
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

#### 7. npy_reader.py
Print data array in numpy dump
- Args
  * np - The path to npy file
```
python npy_reader.py -np <npy_file>
```
