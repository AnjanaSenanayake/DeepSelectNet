import sys
import shutil
import os
from tempfile import mkstemp

import click


@click.command()
@click.option('--fastq', '-fq', help='The source file path', type=click.Path(exists=True))
@click.option('--shortner', '-shrt', default=False, help='Fastq shortner(default=True)', type=bool)
@click.option('--concat', '-cat', default=False, help='Fastq shortner(default=False)', type=bool)
def main(fastq, shortner, concat):
    # Create temp file
    fh, abs_path = mkstemp()
    if shortner:
        with os.fdopen(fh, 'w') as new_file:
            with open(fastq) as old_file:
                prev_line = ''
                for line in old_file:
                    if prev_line != '' and prev_line[0] == '@':
                        subst = line[0:300] + '\n'
                        new_file.write(line.replace(line, subst))
                    elif prev_line != '' and prev_line[0] == '+':
                        subst = line[0:300] + '\n'
                        new_file.write(line.replace(line, subst))
                    else:
                        new_file.write(line.replace(line, line))
                    prev_line = line
        # Copy the file permissions from the old file to the new file
        shutil.copymode(fastq, abs_path)
        # Remove original file
        os.remove(fastq)
        # Move new file
        shutil.move(abs_path, fastq)
    elif concat:
        with os.fdopen(fh, 'w') as new_file:
            root, _, files = next(os.walk(fastq))
            file_count = 0
            for file in files:
                if file.endswith(".fastq"):
                    line_count = 0
                    path = fastq + '/' + file
                    with open(path) as old_file:
                        lines = old_file.readlines()
                        for line in lines:
                            new_file.write(line.strip())
                            new_file.write('\n')
                            line_count += 1
                file_count += 1
                print('{}/{}'.format(file_count, len(files)))
        # Copy the file permissions from the old file to the new file
        shutil.copymode(fastq, abs_path)
        # Move new file
        shutil.move(abs_path, 'fastq_list.fastq')


if __name__ == '__main__':
    main()
