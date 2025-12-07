import glob, os
import labelImg
label_dir = r"D:\GPT\meter-reading-3\train\labels"
class_ids = set()

for f in glob.glob(os.path.join(label_dir, "*.txt")):
    with open(f) as fh:
        for line in fh:
            parts = line.strip().split()
            if parts:
                class_ids.add(parts[0])

print("Class IDs found in labels:", class_ids)
