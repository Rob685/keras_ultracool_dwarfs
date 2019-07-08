import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""This is a test file for Atom. -Rob"""

x = np.arange(0, 2.0*np.pi, 0.001)
y = np.sin(x)
z = np.cos(x)
w = np.log(np.sin(x))

df = pd.DataFrame({'angle': x, 'sine(x)': y, 'cos(x)': z, 'log(x)': w})

print(df)

plt.plot(df['angle'], df['sine(x)'], 'b--', label='sin(x)')
plt.plot(df['angle'], df['cos(x)'], 'r--', label='cos(x)')
# plt.plot(df['angle'],df['log(x)'])
plt.xlabel('angle')
plt.ylabel('output')
plt.legend()
plt.show()

plt.plot(df['angle'], df['log(x)'], 'k--', label='log(x)')
plt.xlabel('angle')
plt.ylabel('output')
plt.legend()
plt.show()
