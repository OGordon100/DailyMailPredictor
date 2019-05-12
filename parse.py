from itertools import product

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

lowercase_text = []
one_hot = []
pub_day = []
pub_month = []
pub_year = []

days = np.arange(1, 32)
months = np.arange(1, 13)
years = np.arange(1996, 2019)

print("Reading Headlines:")
for year, month, day in tqdm(product(years, months, days), total=len(years) * len(months) * len(days)):
    url = "https://www.dailymail.co.uk/home/sitemaparchive/day_%04d%02d%02d.html" % (year, month, day)

    # Open page
    r = requests.get(url)
    if r.status_code != 200:
        continue
    soup = BeautifulSoup(r.text, 'html.parser')
    titles = soup.find("ul", {"class": "archive-articles debate link-box"}).find_all("a")

    # If history on that day
    if len(titles) != 0:
        full_title = [title.text for title in titles]

        for title in full_title:
            # Store info only for titles containing fully capitalised words
            split_title = title.split()
            capitalised_vector = [int(t.isupper()) for t in title.split()]

            if sum(capitalised_vector) > 0:
                lowercase_text.append(title.lower())
                one_hot.append(np.nonzero(capitalised_vector)[0])
                pub_day.append(day)
                pub_month.append(month)
                pub_year.append(year)

# Turn into dframe and save
dframe = pd.DataFrame(
    {"headline": lowercase_text, "capitalised_index": one_hot,
     "pub_day": pub_day, "pub_month": pub_month, "pub_year": pub_year})

print("Saving to CSV")
dframe.to_csv("data.csv")
