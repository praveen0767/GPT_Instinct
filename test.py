# scripts/extract_crops_from_labels.py
import os, glob, cv2
ROOT="."
IMG_DIR="yolo_data/train/images"
LBL_DIR="yolo_data/train/labels"
OUT="ocr_crops"
CLASS_NAMES = {0:"display",1:"serial",2:"kwh",3:"md_kw",4:"decimal"}

os.makedirs(OUT, exist_ok=True)
for idx,f in enumerate(glob.glob(os.path.join(IMG_DIR,"*.*"))):
    base = os.path.splitext(os.path.basename(f))[0]
    img=cv2.imread(f)
    h,w = img.shape[:2]
    lbl_file = os.path.join(LBL_DIR, base + ".txt")
    if not os.path.exists(lbl_file):
        continue
    for line in open(lbl_file):
        if not line.strip(): continue
        parts=line.strip().split()
        cls=int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        x1 = int((xc - bw/2)*w)
        x2 = int((xc + bw/2)*w)
        y1 = int((yc - bh/2)*h)
        y2 = int((yc + bh/2)*h)
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        od = os.path.join(OUT, CLASS_NAMES.get(cls,str(cls)))
        os.makedirs(od, exist_ok=True)
        cv2.imwrite(os.path.join(od, f"{base}_{cls}.jpg"), crop)
print("Crops written to", OUT)
