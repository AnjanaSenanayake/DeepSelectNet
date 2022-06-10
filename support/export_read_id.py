import os
import sys
from ont_fast5_api.fast5_interface import get_fast5_file
import click


@click.command()
@click.option('--fast5', '-f5', help='path to fast5 file', type=click.Path(exists=True))
@click.option('--output', '-o', help='path to output file', type=click.Path(exists=False))
def main(fast5, output):
    for root, dirs, files in os.walk(fast5):
        for file in files:
            if file.endswith(".fast5"):
                with get_fast5_file(root + '/' + file, mode="r") as f5:
                    for read in f5.get_reads():
                        with open(output, "a") as read_file:
                            read_file.write(read.read_id + '\n')
                            print(read.read_id)
    read_file.close()


if __name__ == '__main__':
    main()
