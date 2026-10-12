"""Deteksi SAM 3 pada potongan, dengan cache dan pemetaan polygon ke foto asli."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

from PIL import Image
from review_preview import polygon_boxes


def tile_starts(length, tile_size, overlap):
    if length <= tile_size:
        return [0]
    steps = math.ceil((length-tile_size) / (tile_size*(1-overlap)))
    return [round(index*(length-tile_size)/steps) for index in range(steps+1)]


def map_tile_predictions(response, region, image_size):
    left, top, right, bottom = region
    tile_size = (right-left, bottom-top)
    # Validasi skema dan koordinat dalam ukuran potongan sebelum pemetaan.
    boxes = polygon_boxes(response, tile_size)
    mapped = []
    clipped = 0
    for prediction, detection in zip(response['predictions'], boxes):
        x1, y1, x2, y2 = detection['box']
        # Abaikan kandidat yang terpotong pada batas internal; overlap memberi
        # kesempatan deteksi utuh pada potongan tetangga. Batas foto tetap boleh.
        if ((left > 0 and x1 <= 1) or (top > 0 and y1 <= 1) or
            (right < image_size[0] and x2 >= tile_size[0]-1) or
            (bottom < image_size[1] and y2 >= tile_size[1]-1)):
            clipped += 1
            continue
        prediction = copy.deepcopy(prediction)
        for ring in prediction['vertices']:
            for point in ring:
                point['x'] += left
                point['y'] += top
        mapped.append(prediction)
    return mapped, clipped


def infer_tiles(project, image_path, directory, prompt, confidence, tile_size,
                overlap, save, key, stage, requester=None):
    if requester is None:
        from preview_requests import PreviewRequester
        requester = PreviewRequester()
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    regions = [(x, y, min(x+tile_size, image.width), min(y+tile_size, image.height))
               for y in tile_starts(image.height, tile_size, overlap)
               for x in tile_starts(image.width, tile_size, overlap)]
    settings = {'model': 'sam3-polygon', 'prompt': prompt, 'confidence': confidence,
                'tile_size': tile_size, 'overlap': overlap,
                'image_sha256': hashlib.sha256(image_path.read_bytes()).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:20]
    cache = directory / 'tiles' / fingerprint
    cache.mkdir(parents=True, exist_ok=True)
    save(cache / 'settings.json', settings, key)
    predictions = []
    tile_stats = []
    clipped = 0
    print(f'Deteksi per potongan: {len(regions)} potongan; ukuran {tile_size}px.', flush=True)
    for index, region in enumerate(regions, 1):
        started = time.monotonic()
        stage(f'deteksi potongan {index}/{len(regions)}')
        response_path = cache / f'{index:03d}.json'
        if response_path.is_file():
            response = json.loads(response_path.read_text(encoding='utf-8'))
            origin = 'cache'
        else:
            crop_path = cache / f'{index:03d}.jpg'
            image.crop(region).save(crop_path, quality=95)
            response = requester.preview(project, crop_path.resolve(), prompt, confidence)
            # Jangan menyimpan blockErrors sebagai cache keberhasilan.
            polygon_boxes(response, (region[2]-region[0], region[3]-region[1]))
            save(response_path, response, key)
            origin = 'API'
        mapped, edge_count = map_tile_predictions(response, region, image.size)
        predictions.extend(mapped)
        clipped += edge_count
        elapsed = time.monotonic()-started
        tile_stats.append({'tile':index,'origin':origin,'raw_predictions':len(response['predictions']),
                           'mapped_predictions':len(mapped),'edge_skipped':edge_count,
                           'elapsed_seconds':round(elapsed,2)})
        print(f'Potongan {index}/{len(regions)} ({origin}): '
              f'{len(response["predictions"])} dari model, {len(mapped)} diteruskan, '
              f'{edge_count} terpotong dilewati; {elapsed:.1f} detik.', flush=True)
    return {'model': 'sam3-polygon', 'predictions': predictions,
            'summary': {'totalDetections': len(predictions), 'tiles': len(regions),
                        'clippedInternalEdgeSkipped': clipped,
                        'rawDetections':sum(t['raw_predictions'] for t in tile_stats)},
            'tile_stats':tile_stats,
            'tiling': {'tile_size': tile_size, 'overlap': overlap, 'image_size': list(image.size)}}
