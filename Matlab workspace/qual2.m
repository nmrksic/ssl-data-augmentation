function [acc, err] = qual2(p, r, N1, N)

err = N1 * (p + r - 2 * p * r) / (N * p)

acc = 1-err