"""Auto-label ikan kecil melalui Roboflow. Python 3.10+; lihat PETUNJUK.txt."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


def positions(length, tile_size, overlap):
    if length <= 0 or tile_size < 32 or not 0 <= overlap < 0.8:
        raise ValueError("Ukuran gambar/tile atau overlap tidak valid.")
    if length <= tile_size:
        return [0]
    step = max(1, round(tile_size * (1 - overlap)))
    return sorted(set([*range(0, length - tile_size + 1, step), length - tile_size]))


def tile_regions(width, height, tile_size, overlap):
    return [(x, y, min(x + tile_size, width), min(y + tile_size, height))
            for y in positions(height, tile_size, overlap)
            for x in positions(width, tile_size, overlap)]


def map_prediction(pred, region, image_size, response_size, confidence, class_name):
    """Koordinat keluaran model -> tile -> gambar asli, tanpa mengubah resolusi."""
    if pred.get("class") != class_name:
        return None
    try:
        score, x, y, w, h = [float(pred[k]) for k in
                              ("confidence", "x", "y", "width", "height")]
    except (KeyError, ValueError, TypeError):
        return None
    if not all(math.isfinite(v) for v in (score, x, y, w, h)):
        return None
    if not confidence <= score <= 1 or w <= 0 or h <= 0:
        return None
    ox, oy, ex, ey = region
    rw, rh = response_size
    if rw <= 0 or rh <= 0:
        raise ValueError("Resolusi hasil inference tidak valid.")
    sx, sy = (ex - ox) / rw, (ey - oy) / rh
    iw, ih = image_size
    x1 = max(ox, min(iw, ox + (x - w / 2) * sx))
    y1 = max(oy, min(ih, oy + (y - h / 2) * sy))
    x2 = min(ex, iw, ox + (x + w / 2) * sx)
    y2 = min(ey, ih, oy + (y + h / 2) * sy)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return {"box": [x1, y1, x2, y2], "confidence": score, "class": class_name}


def iou(a, b):
    ax, ay, bx, by = a
    cx, cy, dx, dy = b
    area = max(0, min(bx, dx) - max(ax, cx)) * max(0, min(by, dy) - max(ay, cy))
    union = (bx - ax) * (by - ay) + (dx - cx) * (dy - cy) - area
    return area / union if union > 0 else 0


def deduplicate(detections, threshold):
    kept = []
    for d in sorted(detections, key=lambda p: p["confidence"], reverse=True):
        if not any(iou(d["box"], k["box"]) > threshold for k in kept):
            kept.append(d)
    return kept


def sharpness(image, box):
    """Heuristik variance Laplacian. Bukan bukti bahwa objek benar-benar ikan."""
    import numpy as np
    x1, y1, x2, y2 = box
    crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    if min(gray.shape) < 3:
        return 0.0
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2]
           + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
    return float(lap.var())


def voc_xml(name, size, detections, one_based=False):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = name
    sz = ET.SubElement(root, "size")
    for key, value in zip(("width", "height", "depth"), (*size, 3)):
        ET.SubElement(sz, key).text = str(value)
    for d in detections:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = d["class"]
        ET.SubElement(obj, "difficult").text = "0"
        box = ET.SubElement(obj, "bndbox")
        x1, y1, x2, y2 = d["box"]
        values = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
        if one_based:
            # Roboflow mengurangi 1 ketika mengimpor batas VOC. Gunakan batas
            # 1..ukuran foto agar kotak interior tidak bergeser -1 piksel.
            values = tuple(max(1, min(bound, value + 1))
                           for value, bound in zip(values, (size[0], size[1], size[0], size[1])))
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), values):
            ET.SubElement(box, key).text = str(value)
    return ET.tostring(root, encoding="unicode")


def has_manual_annotations(details):
    """None berarti status tidak tersedia; jangan menulis jika status tidak jelas."""
    known = False
    for key in ("annotations", "labels"):
        if key in details:
            known = True
            if details[key]:
                return True
    return False if known else None


def submit_predictions(project, image_id, xml_path, protected_ids):
    if image_id in protected_ids:
        return {"status": "skipped", "reason": "Foto dilindungi: label manual dipertahankan."}
    state = has_manual_annotations(project.image(image_id))
    if state is not False:
        return {"status": "skipped", "reason": "Ada label lama atau status label tidak tersedia."}
    result = project.save_annotation(
        annotation_path=str(xml_path), image_id=image_id,
        is_prediction=True, annotation_overwrite=False, add_to_dataset=False,
    )
    response = result[0]
    if response.get("warn"):
        return {"status": "skipped", "reason": str(response["warn"])}
    if not response.get("success"):
        raise RuntimeError("Server belum mengonfirmasi pengiriman prediksi.")
    return {"status": "sent", "image_id": image_id}


def detect(image, client, cfg):
    regions = tile_regions(*image.size, cfg["tile_size"], cfg["overlap"])
    candidates, rejected_blur = [], 0
    batch_size = cfg["tile_batch_size"]
    for start in range(0, len(regions), batch_size):
        chunk = regions[start:start + batch_size]
        tiles = [image.crop(r) for r in chunk]
        responses = client.infer(tiles, model_id=cfg["model_id"])
        if isinstance(responses, dict) and len(chunk) == 1:
            responses = [responses]
        if not isinstance(responses, list) or len(responses) != len(chunk):
            raise RuntimeError("Jumlah hasil inference berbeda dari jumlah tile.")
        for tile, region, response in zip(tiles, chunk, responses):
            preds = response.get("predictions")
            if not isinstance(preds, list):
                raise RuntimeError("Model harus menghasilkan bounding box object detection.")
            dims = response.get("image", {})
            resp_size = (dims.get("width", tile.width), dims.get("height", tile.height))
            for p in preds:
                d = map_prediction(p, region, image.size, resp_size,
                                   cfg["confidence"], cfg["class_name"])
                if d is not None:
                    candidates.append(d)
        print(f"  Tile {min(start + batch_size, len(regions))}/{len(regions)}", flush=True)
    candidates = deduplicate(candidates, cfg["nms_iou"])
    accepted = []
    for d in candidates:
        d["sharpness"] = sharpness(image, d["box"])
        if d["sharpness"] < cfg["min_sharpness"]:
            rejected_blur += 1
        else:
            accepted.append(d)
    return accepted, rejected_blur


def load_config(path):
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if not re.fullmatch(r"[^/\s]+/[0-9]+", cfg.get("model_id", "")):
        raise ValueError("Belum ada Model ID. Latih model ikan, lalu isi model_id, misalnya sipandu-tambak/1.")
    for key in ("confidence", "nms_iou"):
        if not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} harus antara 0 dan 1.")
    positions(1000, cfg["tile_size"], cfg["overlap"])
    if cfg["tile_batch_size"] < 1 or cfg["min_sharpness"] < 0:
        raise ValueError("Batch size atau threshold blur tidak valid.")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--images", type=Path, required=True, help="Folder foto asli; subfolder ikut diproses")
    parser.add_argument("--ids", type=Path, help="JSON mapping path relatif -> image ID Roboflow")
    parser.add_argument("--output", type=Path, default=Path("hasil-autolabel"))
    parser.add_argument("--send", action="store_true", help="Kirim prediksi ke image ID yang belum berlabel")
    args = parser.parse_args()
    cfg = load_config(args.config)
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key:
        raise ValueError("Isi environment variable ROBOFLOW_API_KEY di komputer/Colab Anda.")
    if args.send and args.ids is None:
        raise ValueError("Mode --send memerlukan --ids agar prediksi menuju foto yang tepat.")
    if not args.images.is_dir():
        raise ValueError("Folder foto tidak ditemukan.")
    from PIL import Image, ImageDraw, ImageOps
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=key)
    client.configure(InferenceConfiguration(
        confidence_threshold=cfg["confidence"], disable_active_learning=True,
        max_batch_size=1, max_concurrent_requests=4,
    ))
    project = None
    if args.send:
        from roboflow import Roboflow
        project = Roboflow(api_key=key).workspace(cfg["workspace"]).project(cfg["project"])
    ids = json.loads(args.ids.read_text(encoding="utf-8-sig")) if args.ids else {}
    files = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        raise ValueError("Tidak ada foto JPG/PNG/WebP di folder.")
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for source in files:
        relative = source.relative_to(args.images).as_posix()
        image_id = ids.get(relative)
        print(relative, flush=True)
        with Image.open(source) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        detections, blurry = detect(image, client, cfg)
        ident = hashlib.sha256(relative.encode()).hexdigest()[:10]
        basename = f"{source.stem}-{ident}"
        xml_path = args.output / f"{basename}.xml"
        xml_path.write_text(voc_xml(source.name, image.size, detections), encoding="utf-8")
        payload = {"file": relative, "image_size": list(image.size),
                   "model_id": cfg["model_id"], "detections": detections, "rejected_blur": blurry}
        (args.output / f"{basename}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        preview = image.copy()
        draw = ImageDraw.Draw(preview)
        for d in detections:
            draw.rectangle(d["box"], outline="#00ff00", width=2)
        preview.save(args.output / f"{basename}-preview.jpg", quality=94)
        outcome = {"status": "preview"}
        if args.send:
            if not detections:
                outcome = {"status": "skipped", "reason": "Tidak ada deteksi; tidak membuat negative image."}
            elif not image_id:
                outcome = {"status": "skipped", "reason": "Image ID belum tersedia di mapping."}
            else:
                outcome = submit_predictions(project, image_id, xml_path, cfg["protected_image_ids"])
        payload["upload"] = outcome
        report.append(payload)
        (args.output / "laporan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  {len(detections)} kandidat; {blurry} dilewati oleh filter blur; {outcome['status']}", flush=True)
    print(f"Selesai. Hasil: {args.output.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # SDK errors may contain URLs with credentials: do not print the raw exception.
        if isinstance(error, (ValueError, FileNotFoundError, ModuleNotFoundError)):
            message = str(error).replace(os.environ.get("ROBOFLOW_API_KEY") or "\0", "[REDACTED]")
        else:
            message = f"{type(error).__name__}: proses dihentikan. Periksa model, izin API, dan koneksi."
        print(f"Gagal: {message}", file=sys.stderr)
        sys.exit(1)
