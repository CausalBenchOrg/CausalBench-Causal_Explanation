import atexit
import os
import shutil
import tempfile
from urllib.parse import urlparse

import requests


def download_zip_from_url(url, download_dir):
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename or not filename.endswith('.zip'):
            filename = f"downloaded_{abs(hash(url)) % 10000:04d}.zip"
        
        filepath = os.path.join(download_dir, filename)
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return filepath
        
    except Exception as e:
        print(f"Error downloading {url}: {e}")


def fetch_zip_files(zip_urls, download_dir):
    downloaded_files = []
    
    os.makedirs(download_dir, exist_ok=True)
    atexit.register(lambda: shutil.rmtree(download_dir))
    
    for url in sorted(zip_urls):
        filepath = download_zip_from_url(url, download_dir)
        if filepath:
            downloaded_files.append(filepath)
    
    downloaded_files.sort()
     
    print(f"Successfully downloaded {len(downloaded_files)}/{len(zip_urls)} files")

    return downloaded_files


def download_files(zip_urls):
    if zip_urls:
        print(f"Fetching {len(zip_urls)} ZIP files from URLs...")
        
        download_dir = os.path.join(tempfile.gettempdir(), "causal_analysis_fixed")
        print(f"Download directory: {download_dir}")
        
        downloaded_files = fetch_zip_files(zip_urls, download_dir)

        return download_dir, downloaded_files
    else:
        print("No URLs provided")
