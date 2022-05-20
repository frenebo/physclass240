import numpy as np

def make_data():
    N = 10000
    a = 0.5
    z_i = np.random.rand(N) * 2 - 1

    # y_i = a * z_i
    x_i = z_i + np.random.randn(N)
    z_i = z_i * a + np.random.randn(N)

if __name__ == "__main__":
    make_data()