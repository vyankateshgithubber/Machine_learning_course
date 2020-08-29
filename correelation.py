# Import libraries
import pandas as pd
import matplotlib.pyplot as plt


# Read the corr.csv file
f = pd.read_csv('03 - corr.csv')
# covert data into float
f['t0'] = pd.to_numeric(f['t0'],downcast='float')

# plot ACF
plt.acorr(f['t0'],maxlags=10)
plt.show()

# create the shifted values using dataset

t_1 = f['t0'].shift(+1).to_frame()
t_2 = f['t0'].shift(+2).to_frame()

print(t_1)
print(t_2)
