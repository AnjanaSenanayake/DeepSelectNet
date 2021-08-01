import shutil
import os
import click


@click.command()
@click.option('--source', '-src', help='The source directory path', type=click.Path(exists=True))
@click.option('--destination_1', '-dest1', help='The destination 1 directory path', type=click.Path())
@click.option('--split_1', '-sp1', help='The destination 1 split size', type=click.Path())
@click.option('--destination_2', '-dest2', help='The destination 2 directory path', type=click.Path())
@click.option('--split_2', '-sp2', help='The destination 2 split size', type=click.Path())
def main(source, destination_1, split_1, destination_2, split_2):
    _, _, dir_files = next(os.walk(source))

    if split_1:
        split_1 = int(split_1)
    if split_2:
        split_2 = int(split_2)

    dest1_files = dir_files[0:split_1]
    if split_2 and split_2 > 0:
        dest2_files = dir_files[split_1:(split_1 + split_2)]
    else:
        dest2_files = dir_files[split_1:]

    for dest_file in dest1_files:
        shutil.copy(source + '/' + dest_file, destination_1)
        print(dest_file + ' is copying to ' + destination_1)

    for dest_file in dest2_files:
        shutil.copy(source + '/' + dest_file, destination_2)
        print(dest_file + ' is copying to ' + destination_2)


if __name__ == '__main__':
    main()
