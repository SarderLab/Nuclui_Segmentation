import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

IMG_EXTS = {".tif"}

# Return all polygons (list of list of (x,y)) in an XML file.
def load_vertices(xml_path: Path):
    root = ET.parse(str(xml_path)).getroot()
    polygons = []
    for ann in root.findall(".//Annotation"):
        for region in ann.findall(".//Region"):
            verts = region.findall(".//Vertex")
            if not verts:
                continue
            pts = [(int(round(float(v.get("X")))), int(round(float(v.get("Y"))))) for v in verts]
            polygons.append(pts)
    return polygons

def find_image_for_stem(image_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None

def save_binary_mask(image_path: Path, polygons, out_path: Path):
    im = Image.open(image_path)
    W, H = im.size
    mask = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(mask)
    for pts in polygons:
        draw.polygon(pts, fill=255)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Generate binary glom masks using glom image patches + patch-level glom annotations.")
    ap.add_argument("--xml_dir", required=True, help="Directory with glom annotations.")
    ap.add_argument("--image_dir", required=True, help="Directory with original glom image patches.")
    ap.add_argument("--out_dir", required=True, help="Output directory for generated masks.")
    args = ap.parse_args()

    xml_dir = Path(args.xml_dir)
    img_dir = Path(args.image_dir)
    out_root = Path(args.out_dir)

    xmls = sorted(xml_dir.glob("*.xml"))
    if not xmls:
        raise FileNotFoundError(f"[ERROR] No XML files found in {xml_dir}")

    total = 0
    for xml in xmls:
        stem = xml.stem
        img_path = find_image_for_stem(img_dir, stem)
        if not img_path:
            print(f"[ERROR] No matching image for {stem}")
            continue
        polygons = load_vertices(xml)
        if not polygons:
            print(f"[ERROR] No regions found in {xml.name}")
            continue

        out_path = out_root / f"{stem}_mask.png"
        save_binary_mask(img_path, polygons, out_path)
        total += 1
        print(f"{xml.name} -> {out_path.name}")

    print(f"\nDone. Generated {total} mask(s).")

if __name__ == "__main__":
    main()