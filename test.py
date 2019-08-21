import numpy as np
from scipy.stats import norm

mask = np.zeros((672,672))
x = np.linspace(norm.ppf(0.01),norm.ppf(0.5), 672)
target = (336,336)

for r in range(672):
	for c in range(672):
		mask[r,c] = norm.pdf(x[671 - abs(r - target[0])]) * norm.pdf(x[671 - abs(c - target[1])])
		print mask[r,c]
	print("-------------------------------------------")