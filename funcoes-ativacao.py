import numpy as np


# transfer functions
def stepFunction(soma):
    if soma >= 1:
        return 1
    return 0


def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))


def tahnFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))


def reluFunction(soma):
    if soma >= 0:
        return soma
    return 0


def linearFunction(soma):
    return soma


def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()


print(stepFunction(2.1))
print(sigmoidFunction(2.1))
print(tahnFunction(2.1))
print(reluFunction(2.1))
print(linearFunction(2.1))
values = [2.0, 2.0, 2.0]
print(softmaxFunction(values))
