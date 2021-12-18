import click

F1_LEN = 0
F2_LEN = 0
F1_COUNT = 0
F2_COUNT = 0
MATCHED = 0
UNMATCHED = 0


def line_that_contain(line_f1, f2_lines):
    global MATCHED, UNMATCHED, F2_COUNT
    F2_COUNT = 0
    for line_f2 in f2_lines:
        F2_COUNT += 1
        if line_f1 == line_f2.strip():
            MATCHED = MATCHED + 1
            break
        print('{}/{} in {}/{} Matched: {}'.format(F2_COUNT, F2_LEN, F1_COUNT, F1_LEN, MATCHED))


@click.command()
@click.option('--file1', '-f1', help='The source file path', type=click.Path(exists=True))
@click.option('--file2', '-f2', help='The destination file path', type=click.Path(exists=True))
def main(file1, file2):
    global F1_COUNT, F1_LEN, F2_LEN
    with open(file1, 'rb') as file_1:
        with open(file2, 'rb') as file_2:
            f1_lines = file_1.readlines()
            f2_lines = file_2.readlines()
            F1_LEN = len(f1_lines)
            F2_LEN = len(f2_lines)
            for line_f1 in f1_lines:
                F1_COUNT += 1
                line_that_contain(line_f1.strip(), f2_lines)


if __name__ == '__main__':
    main()
