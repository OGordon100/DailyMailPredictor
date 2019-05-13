import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import pandas as pd

data = pd.read_csv("data.csv")
data.pub_date = pd.to_datetime(data.pub_date)
data.capitalised_index = [list(map(int, bodge.replace("[", "").replace("]", "").split())) for bodge in
                          data.capitalised_index.str.replace("'", "")]

data = data.set_index('pub_date', drop=True)
data.capitalised_index.str.len().resample("M").mean().plot(style='x')
plt.title("Rabidness Over Time")
plt.xlabel("Date")
plt.ylabel("Mean Number of Capitalised Words Per Month")

plt.figure()
data.headline.str.count("diana").resample("Y").sum().plot(style='x')
plt.title("Obsession with Diana")
plt.xlabel("Date")
plt.ylabel("Diana Headlines Per Year")
plt.show()