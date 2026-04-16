import os
import shutil
import random

# === 配置路径 ===
base_path = 'D:/pythoncache/dual_modal_insulator_fault_detection/rtdetrv2_pytorch/dataset/visible/visnormal'
img_base = os.path.join(base_path, 'images')
lbl_base = os.path.join(base_path, 'labels')

# === 回滚阶段：将 train/val 文件合并回根目录 ===
def merge_back(subdir):
    for f in os.listdir(os.path.join(img_base, subdir)):
        shutil.move(os.path.join(img_base, subdir, f), img_base)
    for f in os.listdir(os.path.join(lbl_base, subdir)):
        shutil.move(os.path.join(lbl_base, subdir, f), lbl_base)
    shutil.rmtree(os.path.join(img_base, subdir))
    shutil.rmtree(os.path.join(lbl_base, subdir))

for split in ['train', 'val']:
    if os.path.exists(os.path.join(img_base, split)):
        merge_back(split)

# === 图像筛选 & 打乱 ===
image_files = [f for f in os.listdir(img_base) if f.lower().endswith(('.jpg', '.png'))]
random.shuffle(image_files)

split_ratio = 0.8
split_index = int(len(image_files) * split_ratio)
train_files = image_files[:split_index]
val_files = image_files[split_index:]

# === 创建新目录 ===
for split in ['train', 'val']:
    os.makedirs(os.path.join(img_base, split), exist_ok=True)
    os.makedirs(os.path.join(lbl_base, split), exist_ok=True)

# === 重新划分 ===
def move_files(files, split):
    moved_count = 0
    for img in files:
        base = os.path.splitext(img)[0]
        img_src = os.path.join(img_base, img)
        lbl_src = os.path.join(lbl_base, base + '.txt')

        img_dst = os.path.join(img_base, split, img)
        lbl_dst = os.path.join(lbl_base, split, base + '.txt')

        if os.path.exists(img_src) and os.path.exists(lbl_src):
            shutil.move(img_src, img_dst)
            shutil.move(lbl_src, lbl_dst)
            moved_count += 1
        else:
            print(f"⚠️ 跳过缺失：{img} 或 {base}.txt")

    print(f"✅ [{split}] 划分完成，共 {moved_count} 张图像。")

move_files(train_files, 'train')
move_files(val_files, 'val')

print(f"\n📊 总图像数：{len(image_files)}")
print(f"📁 训练集：{len(train_files)}")
print(f"📁 验证集：{len(val_files)}")
