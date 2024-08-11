import os
from tqdm import tqdm
import requests

url = {
    'owl_v2_b16': 'https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b',
    'owl_v2_l14': 'https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-l14-1008-st-ngrams-ft-lvisbase-ens-cold-weight-04_8ca674c'
}

def download(url, path):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)


def load_checkpoint(name='owl_v2_b16'):
    this_dir = os.path.dirname(os.path.realpath(__file__))

    checkpoint_path = os.path.join(this_dir, f'{name}.cpt')

    if not os.path.exists(checkpoint_path):
        download(url[name], checkpoint_path)

    return checkpoint_path

