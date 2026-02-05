import numpy as np
import matplotlib.pyplot as plt

an_gradients = np.array([
    -2.4585865321569145e-04,
    -3.9259056211449206e-04,
    -4.7801239998079836e-04,
    -5.1525101298466320e-04,
    -5.1075784722343090e-04,
    -4.6998771722428500e-04,
    -3.9587778155691920e-04,
    -2.9158205143176020e-04,
    -1.6087359108496457e-04,
    9.9369742656563180e-07
], dtype=np.float64)

loss_values = np.array([
    1.9467296078801155e-03,
    1.2773488415405154e-03,
    8.8194472482427950e-04,
    6.0072966152802110e-04,
    4.1963582043536010e-04,
    2.8264205320738256e-04,
    1.9572641758713870e-04,
    1.3291477807797492e-04,
    1.0154065239476040e-04,
    9.4755618192721160e-05
], dtype=np.float64)

indices = np.arange(1, len(an_gradients) + 1)

for title, y, ylabel in [
    ("AN Gradient values", an_gradients, "AN gradient"),
    ("Loss values", loss_values, "Loss value"),
]:
    plt.figure()
    plt.plot(indices, y, marker='o')
    plt.xlabel("Sample index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plt.figure()
plt.plot(loss_values, an_gradients, marker='o')
plt.xlabel("Loss value")
plt.ylabel("AN gradient")
plt.title("AN Gradient vs Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
