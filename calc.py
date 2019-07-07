import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,2.0*np.pi,0.001)
y = np.sin(x)
z = np.cos(x)

df = pd.DataFrame({'angle':x,'sine(x)':y,'cos(x)':z})

print(df)

plt.plot(df['angle'],df['sine(x)'],'b--',label='sin(x)')
plt.plot(df['angle'],df['cos(x)'],'r--',label='cos(x)')
plt.xlabel('angle')
plt.ylabel('trig_func')
plt.legend()
plt.show()
