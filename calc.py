import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.arange(0,2.0*np.pi,0.001)
y = np.sin(x)

df = pd.DataFrame({'angle':x,'sine(x)':y})

print(df)

plt.plot(df['angle'],df['sine(x)'],'k--')
plt.xlabel('angle')
plt.ylabel('sine(x)')
plt.show()
