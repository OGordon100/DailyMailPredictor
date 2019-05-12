import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

database_name = "data_no_short_words.csv"
lowercase_text = []
one_hot = []
pub_date = []

if os.path.isfile(database_name):
    start_date = pd.read_csv(database_name).tail(1).pub_date.values[0] + pd.Timedelta(days=1)
else:
    start_date = '1996-01-01'
end_date = pd.datetime.today()

print("Reading Archive Webpages:")
for date in tqdm(pd.date_range(start_date, end_date)):
    url = "https://www.dailymail.co.uk/home/sitemaparchive/day_%04d%02d%02d.html" % (date.year, date.month, date.day)

    # Open page
    r = requests.get(url)
    if r.status_code != 200:
        continue
    soup = BeautifulSoup(r.text, 'html.parser')
    titles = soup.find("ul", {"class": "archive-articles debate link-box"}).find_all("a")

    # Store info only for titles containing fully capitalised words
    if len(titles) != 0:
        full_title = [title.text for title in titles]
        for title in full_title:
            capitalised_vector = [int(t.isupper()) for t in title.split() if len(t) > 3]
            if sum(capitalised_vector) > 0:
                lowercase_text.append(title.lower())
                one_hot.append(np.nonzero(capitalised_vector)[0])
                pub_date.append(date)

# Turn into dframe and save
print("Saving to CSV")
dframe = pd.DataFrame(
    {"headline": lowercase_text, "capitalised_index": one_hot,
     "pub_date": pub_date})

if os.path.isfile(database_name):
    dframe.to_csv(database_name, mode='a', header=False, index=False)
else:
    dframe.to_csv(database_name, index=False)
