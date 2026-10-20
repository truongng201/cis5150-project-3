"""Export Project 3 tables and plots; optionally collect two mouse examples."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from interpatxy import interpatxy
from get_points import get_points

# Test points
x: list[np.ndarray] = [np.zeros(0) for _ in range(6)]
y: list[np.ndarray] = [np.zeros(0) for _ in range(6)]
x[0] = np.array([3.6942,1.3690,2.9865,5.8509,8.1929,8.2098,6.8281])
y[0] = np.array([1.2144,3.5925,7.3933,7.9217,6.9665,4.0396,1.5600])
x[1] = np.array([3.9806,2.2789,3.6942,6.8618,7.1820])
y[1] = np.array([2.1087,4.2429,7.0884,6.9461,4.3852])
x[2] = np.array([4.2334,1.0826,1.3016,4.9579,8.2435,4.8062])
y[2] = np.array([1.0315,3.6941,5.6250,7.9624,5.5640,5.8486])
x[3] = np.array([4.6040,2.2283,3.3741,2.1609,7.2494,6.8955,9.1702])
y[3] = np.array([1.3364,1.6616,3.5722,6.8242,8.6535,3.7957,2.8608])
x[4] = np.array([5.4297,5.2275,2.9865,1.4532,2.1778,3.2898,6.8113,9.0691,7.2999,7.2157,9.2713,7.4853,6.4575])
y[4] = np.array([4.6494,1.9055,1.7429,3.8974,8.0030,6.7429,9.1209,7.2917,6.4380,2.8404,2.7795,0.9502,1.3974])
x[5] = np.array([2.7310,1.3599,1.1662,1.9709,4.7876,7.0827,6.2630,4.2809,3.9232,4.9367,8.3048,9.0052,7.6639])
y[5] = np.array([1.1106,2.4716,5.3639,7.3677,8.3129,7.6134,5.3072,5.0614,2.8875,1.0539,0.8648,2.5095,3.7760])


def write_table(out, headers, dat):
    with open(out, 'w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(zip(*dat))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-interactive', action='store_true',
                        help='Export six supplied datasets without mouse input')
    parser.add_argument('--output', type=Path, default=Path('output'))
    args = parser.parse_args()
    for folder in ('part2i', 'part2ii', 'part2iii', 'report', 'code'):
        (args.output / folder).mkdir(parents=True, exist_ok=True)
    for i, (xs, ys) in enumerate(zip(x, y), start=1):
        dx, dy, Bx, By = interpatxy(xs, ys)
        plt.title(f'Natural cubic interpolation — dataset {i}')
        write_table(args.output / 'part2i' / f'{i}.txt',
                    ['dx', 'dy'], [dx, dy])
        write_table(args.output / 'part2ii' / f'{i}.txt',
                    [f'{axis}_{j+1}' for axis in ('Bx', 'By') for j in range(4)],
                    [*Bx.T, *By.T])
        plt.savefig(args.output / 'part2ii' / f'img{i}.png', dpi=160,
                    bbox_inches='tight')
        plt.close('all')
    print(f'Exported 6 datasets to {args.output.resolve()}')
    if args.no_interactive:
        return
    for i in range(1, 3):
        plt.figure()
        xs, ys = get_points()
        plt.close()
        interpatxy(xs, ys)
        plt.title(f'Mouse input — example {i}')
        plt.savefig(args.output / 'part2iii' / f'img{i}.png', dpi=160,
                    bbox_inches='tight')
        plt.show()
        plt.close('all')


if __name__ == '__main__':
    main()
