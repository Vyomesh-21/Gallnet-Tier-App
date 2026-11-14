"""Download the authors' public checkpoint and notebook; do not guess training metadata."""
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import json
import re

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
BASE = 'https://raw.githubusercontent.com/rakeshchandrajoshi/gallbladder_ultrasound/main/'
CHECKPOINT = 'model-031-0.979685-0.992509.keras'
NOTEBOOK = 'gallbladder_ultrasound_self_attention.ipynb'

def fetch(name, output):
    if output.is_file() and output.stat().st_size > 100_000:
        print(f'Already downloaded: {output.name} ({output.stat().st_size:,} bytes)')
        return
    print(f'Downloading {name} from authors\' GitHub repository...')
    request = Request(BASE + name, headers={'User-Agent':'GallbladderResearchApp/1.0'})
    temp = output.with_suffix(output.suffix + '.part')
    try:
        with urlopen(request, timeout=90) as response, temp.open('wb') as f:
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        if temp.stat().st_size < 100_000:
            raise RuntimeError(f'Download unexpectedly small ({temp.stat().st_size} bytes)')
        temp.replace(output)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    print(f'Saved: {output} ({output.stat().st_size:,} bytes)')

def inspect_notebook(path):
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'Could not parse notebook: {exc}')
        return
    print('\nTRAINING METADATA HINTS (inspect these before configuring):')
    keywords = ('class_indices', 'classes=', 'flow_from_directory', 'image_dataset_from_directory', 'rescale=', 'preprocess_input', 'target_size=', 'color_mode=', 'class_names', 'labels=', 'input_shape', 'image_size', 'rescale(')
    matches = 0
    for i, cell in enumerate(data.get('cells', [])):
        source = ''.join(cell.get('source', []))
        for line in source.splitlines():
            if any(k in line for k in keywords):
                print(f'Cell {i}: {line.strip()[:220]}')
                matches += 1
                if matches >= 70:
                    print('(additional matches omitted)')
                    return
    if not matches:
        print('No obvious metadata found; inspect the training notebook manually.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-only', action='store_true')
    args = parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    try:
        fetch(CHECKPOINT, MODELS / CHECKPOINT)
        if not args.checkpoint_only:
            fetch(NOTEBOOK, MODELS / NOTEBOOK)
            inspect_notebook(MODELS / NOTEBOOK)
    except Exception as exc:
        raise SystemExit(f'Download failed: {exc}\nYou can manually download the two files from {BASE}')
    print('\nNext: verify original class index ordering and pixel preprocessing in the notebook.')
    print('Then create models/labels.json and set MODEL_PREPROCESS (zero_one / zero_255 / minus_one_one).')
    print('The app intentionally refuses inference without verified metadata.')
