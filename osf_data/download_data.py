import subprocess
import concurrent.futures
from pathlib import Path

# List of URLs to download
tokens = [
    'hb437', 'bhxpq', 
    'eagwu', 'uyj5e', 
    'wbg5q', 'qfv5k', 
    '3k7p5', 'unj32', 
    'e5vkw', 'zby9q'
]

savepath = Path('/mnt/rradev/osf_data_512px')


def get_url(file_id: str) -> str:
    return f'https://osf.io/{file_id}/download'

def download_file(token):
    url = get_url(token)
    full_save_path = savepath / f"dlprod_512px_{token}.root"
    # Use subprocess to call wget and add name
    subprocess.run(["wget", url, "-O",str(full_save_path)])

# Use a ThreadPoolExecutor to download files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(download_file, tokens)
