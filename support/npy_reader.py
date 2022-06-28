import numpy as np
import click


@click.command()
@click.option('--numpy_array', '-np', help='The numpy file path', type=click.Path(exists=True))
def main(numpy_array):
    np_array = np.load(numpy_array)
    print(np_array)
    print(np_array[0])
    print(np_array.shape)


if __name__ == '__main__':
    main()