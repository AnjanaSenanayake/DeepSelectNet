import sys
import shutil
import os

import click

file_count = 0


@click.command()
@click.option('--src', '-s', help='The source directory path', type=click.Path(exists=True))
@click.option('--dest', '-d', help='The destination directory path', type=click.Path(exists=False))
@click.option('--is_iterative', '-i', default=True, help='Files including child directories', type=bool)
@click.option('--extension', '-ext', default='', help='Extension type to filter', type=str)
@click.option('--count', '-c', default=0, help='The number of files to be copied', type=int)
@click.option('--cut', '-cut', default=False, help='Cut & paste source file', type=bool)
@click.option('--delete', '-del', default=False, help='Delete source file', type=bool)
def main(src, dest, is_iterative, extension, count, cut, delete):
    if not delete and not os.path.exists(dest):
        os.makedirs(dest)

    if is_iterative:
        for root, dirs, files in os.walk(src):
            copy_files(root, files, dest, count, extension, cut, delete)
    else:
        root, dirs, files = next(os.walk(src))
        copy_files(root, files, dest, count, extension, cut, delete)


def copy_files(root, files, dest, count, extension, cut, delete):
    for file in files:
        if extension and file.endswith("." + extension):
            copy(root, dest, file, files, cut, delete)
        else:
            copy(root, dest, file, files, cut, delete)

        if 0 < count == file_count:
            sys.exit()


def copy(root, dest, file, files, cut, delete):
    global file_count
    file_count = file_count + 1
    if cut:
        shutil.move(root + '/' + file, dest)
        print(file + ' is moving to ' + dest + ' ' + str(file_count) + ' remaining ' + str(len(files) - file_count))
    elif delete:
        os.remove(root + '/' + file)
        print(file + ' is deleted ' + ' ' + str(file_count) + ' remaining ' + str(len(files) - file_count))
    else:
        shutil.copy(root + '/' + file, dest)
        print(file + ' is copying to ' + dest + ' ' + str(file_count) + ' remaining ' + str(len(files) - file_count))


if __name__ == '__main__':
    main()
