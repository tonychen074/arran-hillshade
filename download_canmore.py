"""Download Canmore Points data from HES portal."""
import requests
import re
import os
import zipfile

# Step 1: Find download links
url = "https://portal.historicenvironment.scot/downloads/canmore"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
r = requests.get(url, headers=headers)
print("Page status:", r.status_code)

# Find all links
pattern = r'href=["\']([^"\']*)["\']'
all_links = re.findall(pattern, r.text)
for l in all_links:
    if "zip" in l.lower() or "canmore" in l.lower() or "download" in l.lower() or "nrhe" in l.lower():
        print("  ->", l)
