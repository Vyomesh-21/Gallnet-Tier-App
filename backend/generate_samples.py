import os
import io
from pathlib import Path
import numpy as np
from PIL import Image

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
SAMPLES_DIR.mkdir(exist_ok=True)

def generate_ultrasound_phantom(filename, wall_defect=False):
    width, height = 256, 256
    rng = np.random.RandomState(101 if wall_defect else 42)
    
    # Rayleigh speckle noise
    n1 = rng.normal(70, 20, (height, width))
    n2 = rng.normal(50, 18, (height, width))
    speckle = np.sqrt(n1**2 + n2**2)
    
    # Gallbladder lumen (dark / anechoic)
    y, x = np.ogrid[:height, :width]
    cy, cx = height // 2, width // 2
    lumen = ((x - cx)**2) / (65**2) + ((y - cy)**2) / (42**2) <= 1.0
    speckle[lumen] *= 0.12
    
    # Gallbladder wall (hyperechoic / bright)
    wall = (((x - cx)**2) / (72**2) + ((y - cy)**2) / (48**2) <= 1.0) & ~lumen
    
    if wall_defect:
        # Create a focal disruption / perforation defect in wall (upper right quadrant)
        defect = (x > cx + 30) & (y < cy)
        wall = wall & ~defect
        # Extraluminal fluid collection / leakage outside wall
        leak = (((x - (cx + 45))**2) + ((y - (cy - 25))**2) <= 18**2)
        speckle[leak] *= 0.2
        
    speckle[wall] = np.clip(speckle[wall] * 2.3 + 90, 0, 255)
    
    # Acoustic shadow / distal enhancement
    shadow = (x > cx - 20) & (x < cx + 20) & (y > cy + 40)
    speckle[shadow] *= 0.4
    
    img_arr = np.clip(speckle, 0, 255).astype(np.uint8)
    img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
    
    pil = Image.fromarray(img_rgb)
    out_path = SAMPLES_DIR / filename
    pil.save(out_path, format="JPEG", quality=92)
    print(f"Generated sample scan: {out_path} ({out_path.stat().st_size:,} bytes)")

generate_ultrasound_phantom("sample_ultrasound_intact.jpg", wall_defect=False)
generate_ultrasound_phantom("sample_ultrasound_perforation_hazard.jpg", wall_defect=True)

