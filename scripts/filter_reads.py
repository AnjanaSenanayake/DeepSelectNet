import os
import shutil
import click

FAST5_DIR = ''
FAST5_LIST = ''
OUTPUT = ''
FILES_COUNT = 0


def copy_fast5s(root, files, targets):
    global FILES_COUNT
    for file in files:
        path = root + '/' + file
        if file.endswith(".fast5") and file.split('.')[0] in targets:
            FILES_COUNT = FILES_COUNT + 1
            shutil.copy(path, OUTPUT)
            print("Retrieving " + file.split('.')[0] + ' ' + str(FILES_COUNT) + "/" + str(FAST5_LIST))


@click.command()
@click.option('--fast5_files', '-f5', help='path to fast5 files', type=click.Path(exists=True))
@click.option('--target_list', '-tl', help='path to targeted fast5 list file', type=click.Path(exists=True))
@click.option('--output', '-o', help='path to tsv file', type=click.Path(exists=True))
def main(fast5_files, target_list, output):
    global FAST5_DIR, FAST5_LIST, OUTPUT
    FAST5_DIR = fast5_files
    OUTPUT = output

    if not os.path.exists(OUTPUT):
        os.makedirs(OUTPUT)

    f5_list = open(target_list, 'r').read().splitlines()
    FAST5_LIST = len(f5_list)

    for root, dirs, files in os.walk(FAST5_DIR):
        if files:
            copy_fast5s(root, files, f5_list)


if __name__ == '__main__':
    main()
