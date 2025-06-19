#!/usr/bin/env python3
import re

# 读取文件
with open('retinanet/dataloader_dsec_det.py', 'r') as f:
    content = f.read()

# 替换导入
content = content.replace(
    'import skimage.transform',
    '''try:
    import skimage.transform
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    import cv2'''
)

# 替换resize调用
content = re.sub(
    r'image = skimage\.transform\.resize\(image, \(int\(round\(rows\*scale\)\), int\(round\(\(cols\*scale\)\)\)\)\)',
    '''if SKIMAGE_AVAILABLE:
            image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        else:
            image = cv2.resize(image, (int(round(cols*scale)), int(round(rows*scale))))''',
    content
)

# 写回文件
with open('retinanet/dataloader_dsec_det.py', 'w') as f:
    f.write(content)

print("✅ 已修复 dataloader_dsec_det.py")