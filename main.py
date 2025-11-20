import datetime
import time

import numpy as np


def main():
    print("Hello from embodied-exps!")
    test = [
        (1, "Yes", 3.54),
        (2, "No", 7.21),
        (3, "Maybe", 0.12),
        (4, "Definitely", 5.67),
    ]
    idx = [1, 2, 3, 4]
    weights = [0.1, 0.2, 0.3, 0.4]

    test_2 = list(map(np.stack, zip(*test, strict=False))) + [idx, weights]
    print(test_2)
    x = np.array(test_2[2])
    print(x)
    print(f"{time.localtime()}")
    now = datetime.datetime.now()
    print(now)
    time_str = now.strftime("%Y%m%d-%H%M")
    print(time_str)


if __name__ == "__main__":
    main()
