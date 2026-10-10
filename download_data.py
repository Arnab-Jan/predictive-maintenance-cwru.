import os
import urllib.request

os.makedirs('data/raw', exist_ok=True)

urls = {
    "98.mat": "https://engineering.case.edu/sites/default/files/98.mat",   # Normal Baseline
    "108.mat": "https://engineering.case.edu/sites/default/files/108.mat", # Inner Race Fault
    "122.mat": "https://engineering.case.edu/sites/default/files/122.mat", # Ball Fault
    "133.mat": "https://engineering.case.edu/sites/default/files/133.mat"  # Outer Race Fault
}

# Browser headers to prevent CWRU server block
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}

for file_name, url in urls.items():
    dest_path = os.path.join('data/raw', file_name)
    print(f"Downloading {file_name}...")
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        size_kb = os.path.getsize(dest_path) / 1024
        print(f"Success: {file_name} ({size_kb:.1f} KB)")
    except Exception as e:
        print(f"Failed to download {file_name}. Error: {e}")

print("Data acquisition complete.")
