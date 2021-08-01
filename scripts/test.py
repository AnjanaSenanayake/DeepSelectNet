from fast5_research import Fast5, BulkFast5

filename = '../../datasets/covid/multi_fast5/SP1-mapped0.fast5'
# filename = '../../datasets/covid/single_fast5/0/0a4fd1d7-96ff-4ada-9c84-9aeaad21591a.fast5'

with BulkFast5(filename) as fh:
    raw = fh.get_raw()
    print(raw)