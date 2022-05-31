from pathlib import Path
from random import shuffle

import webdataset as wds
from PIL import Image
from tqdm.auto import tqdm

clear = Path('./clear')
watermark = Path('./watermark')

clears = list(clear.glob('./*.jpg'))
watermarks = list(watermark.glob('./*.jpg'))

combined = clears + watermarks
shuffle(combined)

writer = wds.TarWriter('./combined.tar')
for i, p in enumerate(tqdm(combined)):
    k = p.stem
    t = p.parent.name
    sample = {
        '__key__': f'{i:04}',
        'txt': t,
        'jpg': open(p, 'rb').read()
    }
    writer.write(sample)
