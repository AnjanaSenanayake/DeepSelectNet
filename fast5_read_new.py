from fast5_research import Fast5

filename = '../datasets/covid/SP1-raw/SP1-fast5-mapped/single_reads/0/0a89d4a5-0da9-4b98-9446-646e84308167.fast5'

with Fast5(filename) as fh:
    raw = fh.get_read(raw=True)
    channel_meta = fh.channel_meta
    summary = fh.summary()
print('Raw is {} samples long.'.format(len(raw)))
print(channel_meta)
print(raw)
print('Summary {}.'.format(summary))
