import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    u = np.array([[-2, 3]])
    v = np.array([[1, 2]])

    w = np.array([[0, 1]])

    vp = v - u
    wp = w - u

    P = vp.T @ vp / (vp @ vp.T)
    op = wp @ P.T

    o = op + u

    a = (u[0, 1] - v[0, 1]) / (u[0, 0] - v[0, 0])
    b = 0.5 * (u[0, 1] + v[0, 1]) - 0.5 * (u[0, 0] + v[0, 0]) * a

    x = np.array([-4, 5])
    y = a * x + b
    plt.plot(x, y)
    vectors = [u, v, w, vp, wp, op, o]
    vector_names = ["u", "v", "w", "v'", "w'", "p'", "p"]
    plt.quiver([0] * len(vectors), [0] * len(vectors), [p[0, 0] for p in vectors], [p[0, 1] for p in vectors], angles='xy', scale_units='xy', scale=1)
    for xp, yp, text in zip([p[0, 0] for p in vectors], [p[0, 1] for p in vectors], vector_names):
        plt.text(xp, yp, text)
    plt.xlim([-4, 5])
    plt.ylim([-3, 4])
    plt.grid()
    plt.show()