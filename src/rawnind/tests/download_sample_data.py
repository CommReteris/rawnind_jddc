import os
import requests
from pathlib import Path
import json


def download_sample_data(output_dir='src/rawnind/tests/test_data/raw_samples'):
    """Download sample RAW data for testing purposes.
    
    Downloads one small Bayer RAW file from the RawNIND dataset
    for real end-to-end testing.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # For RGB, use existing JPG; no download needed
    rgb_path = 'src/rawnind/tests/test_data/Moor_frog_bl.jpg'
    if os.path.exists(rgb_path):
        print(f'RGB sample available at {rgb_path}')
    else:
        print(f'Warning: RGB sample not found at {rgb_path}')
    
    # For RAW sample, try to fetch from UCLouvain Dataverse
    # RawNIND dataset: https://doi.org/10.14428/DVN/DEQCIM
    dataverse_url = 'https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM'
    
    try:
        print("Fetching dataset metadata from UCLouvain Dataverse...")
        response = requests.get(dataverse_url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        files = data['data']['latestVersion']['files']
        
        # Find a small Bayer file (prefer CR2/ARW files under 10MB)
        target_file = None
        for file_info in files:
            filename = file_info['dataFile']['filename']
            filesize = file_info['dataFile']['filesize']
            
            # Look for Bayer pattern files that are reasonably small
            if ('Bayer' in filename or filename.lower().endswith(('.cr2', '.arw', '.dng'))) and filesize < 10*1024*1024:
                target_file = file_info
                break
        
        if target_file:
            file_id = target_file['dataFile']['id']
            filename = target_file['dataFile']['filename']
            download_url = f'https://dataverse.uclouvain.be/api/access/datafile/{file_id}'
            filepath = Path(output_dir) / filename
            
            if not filepath.exists():
                print(f'Downloading {filename} ({target_file["dataFile"]["filesize"]} bytes)...')
                dl_response = requests.get(download_url, stream=True, timeout=60)
                dl_response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    for chunk in dl_response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            
                print(f'Sample RAW downloaded to {filepath}')
                return str(filepath)
            else:
                print(f'Sample RAW already exists at {filepath}')
                return str(filepath)
        else:
            print('No suitable small RAW sample found in dataset')
            return None
            
    except requests.RequestException as e:
        print(f'Failed to download from dataverse: {e}')
        print('Will use synthetic data for Bayer tests')
        return None
    except (KeyError, json.JSONDecodeError) as e:
        print(f'Failed to parse dataverse response: {e}')
        print('Will use synthetic data for Bayer tests')
        return None


def get_sample_rgb_path():
    """Get path to existing RGB sample image."""
    rgb_path = 'src/rawnind/tests/test_data/Moor_frog_bl.jpg'
    if os.path.exists(rgb_path):
        return rgb_path
    return None


def get_sample_raw_path():
    """Get path to downloaded RAW sample, downloading if needed."""
    output_dir = 'src/rawnind/tests/test_data/raw_samples'
    
    # Check if we already have a RAW file
    if os.path.exists(output_dir):
        for file_path in Path(output_dir).glob('*'):
            if file_path.suffix.lower() in ['.cr2', '.arw', '.dng', '.nef']:
                return str(file_path)
    
    # Try to download
    return download_sample_data(output_dir)


if __name__ == '__main__':
    download_sample_data()
