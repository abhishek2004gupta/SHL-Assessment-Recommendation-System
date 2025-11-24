import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

BASE = "https://www.shl.com/products/product-catalog/?start={}&type={}"

def scrape_page(start, t):
    url = BASE.format(start, t)
    print("Scraping:", url)

    res = requests.get(url)
    if res.status_code != 200:
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table")
    if table is None:
        return []

    rows = table.find_all("tr", attrs={"data-course-id": True})
    items = []

    for r in rows:
        name_tag = r.find("a")
        if not name_tag:
            continue
        name = name_tag.text.strip()
        url = "https://www.shl.com" + name_tag["href"]

        items.append({
            "name": name,
            "url": url,
        })

    return items


def full_scrape():
    all_items = []
    
    # Pre-packaged: type=2 (12 pages approx)
    for start in range(0, 2000, 12):
        items = scrape_page(start, 2)
        if not items:
            break
        all_items.extend(items)

    # Individual tests: type=1 (32 pages approx)
    for start in range(0, 4000, 12):
        items = scrape_page(start, 1)
        if not items:
            break
        all_items.extend(items)

    df = pd.DataFrame(all_items)
    df.drop_duplicates(subset=["url"], inplace=True)
    print("Total scraped:", len(df))
    df.to_csv("shl_catalog_full_details.csv", index=False)

if __name__ == "__main__":
    full_scrape()
