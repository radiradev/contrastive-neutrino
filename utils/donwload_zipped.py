import os 

urls = {
    'data': 'https://files.osf.io/v1/resources/vruzp/providers/osfstorage/5c07dfd565e965001bdf2a7f/?zip=',
    'cluster': 'https://files.osf.io/v1/resources/vruzp/providers/osfstorage/5c427b9a2e57200017fb3aa8/?zip=',
    'particle' : 'https://files.osf.io/v1/resources/vruzp/providers/osfstorage/5c427b39154ce50018dd345a/?zip=', 
}



def download_file(url, savepath):
    os.system(f"wget {url} -O {savepath}")

for key, url in urls.items():
    savepath = f"/mnt/rradev/osf_data_512px/{key}.zip"
    download_file(url, savepath)