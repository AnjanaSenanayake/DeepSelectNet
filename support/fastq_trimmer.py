import shutil
import os
from tempfile import mkstemp
import click


@click.command()
@click.option('--fastq', '-fq', help='The source file path', type=click.Path(exists=True))
def main(fastq):
    # Create temp file
    fh, abs_path = mkstemp()

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


if __name__ == '__main__':
    main()
