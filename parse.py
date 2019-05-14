import os
from itertools import product

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

database_name = "data_no_short_words_newnew.csv"
paper_names = ["daily star", "express", "daily mail"]

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}

lowercase_text = []
one_hot = []
pub_date = []
paper = []

if os.path.isfile(database_name):
    start_date = pd.to_datetime(pd.read_csv(database_name).tail(1).pub_date.values[0]) + pd.Timedelta(days=1)
else:
    start_date = '1996-01-01'
end_date = pd.datetime.today()
date_range = pd.date_range(start_date, end_date)

print("Reading Archive Webpages:")
for paper_name, date in tqdm(product(paper_names, date_range), total=len(paper_names)*len(date_range)):
    if paper_name == "daily mail":
        url = "https://www.dailymail.co.uk/home/sitemaparchive/day_%04d%02d%02d.html" %(date.year, date.month, date.day)
    elif paper_name == "daily star":
        url = "https://www.dailystar.co.uk/sitearchive/%04d/%01d/%01d" %(date.year, date.month, date.day)
    elif paper_name == "express":
        url = "https://www.express.co.uk/sitearchive/%04d/%01d/%01d" %(date.year, date.month, date.day)
    else:
        raise ValueError("Paper not supported!")

    # Open page
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        continue
    soup = BeautifulSoup(r.text, 'html.parser')
    if paper_name == 'daily mail':
        titles = soup.find("ul", {"class": "archive-articles debate link-box"}).find_all("a")
    elif paper_name == 'daily star' or paper_name == 'express':
        titles = soup.find("ul", {"class": "section-list"}).find_all("a")
    else:
        titles = None

    # Store info only for titles containing fully capitalised words
    if len(titles) != 0:
        full_title = [title.text for title in titles]
        for title in full_title:
            capitalised_vector = [int(len(t) > 3 and (t.isupper())) for t in title.split()]
            if sum(capitalised_vector) > 0:
                lowercase_text.append(title.lower())
                one_hot.append(np.nonzero(capitalised_vector)[0])
                pub_date.append(date)
                paper.append(paper_name)

# Turn into dframe and save
print("Saving to CSV")
dframe = pd.DataFrame(
{"paper": paper, "headline": lowercase_text, "capitalised_index": one_hot,
 "pub_date": pub_date})

if os.path.isfile(database_name):
    dframe.to_csv(database_name, mode='a', header=False, index=False)
else:
    dframe.to_csv(database_name, index=False)
