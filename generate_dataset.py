import os
import cv2
import numpy as np
import random

DATASET_DIR = "dataset"
CATEGORIES = {
    "normal": 0,
    "filament_not_deposited": 1,
    "filament_not_adhering": 2,
    "uncertain": 3
}

def create_base_bed(width=400, height=400):
    """Creates a realistic textured 3D printer bed background."""
    # Print bed base color (textured dark gray/black PEI plate)
    bed = np.full((height, width, 3), 35, dtype=np.uint8)
    
    # Add subtle PEI texture noise
    noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
    bed = np.clip(bed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Optional grid lines on print bed (spaced every 50px)
    grid_color = (55, 55, 60)
    for x in range(0, width, 50):
        cv2.line(bed, (x, 0), (x, height), grid_color, 1)
    for y in range(0, height, 50):
        cv2.line(bed, (0, y), (width, y), grid_color, 1)
        
    return bed

def draw_nozzle(img, nozzle_x=200, nozzle_y=120):
    """Draws a realistic brass V6/MK8 3D printer nozzle tip."""
    # Brass color: BGR (30, 180, 220)
    brass_color = (30, 175, 215)
    brass_shadow = (15, 120, 160)
    
    # Upper heat block / nozzle body
    pts_body = np.array([
        [nozzle_x - 35, nozzle_y - 80],
        [nozzle_x + 35, nozzle_y - 80],
        [nozzle_x + 30, nozzle_y - 30],
        [nozzle_x - 30, nozzle_y - 30]
    ], np.int32)
    cv2.fillPoly(img, [pts_body], brass_color)
    cv2.polylines(img, [pts_body], True, brass_shadow, 2)
    
    # Lower conical nozzle tip pointing down to (nozzle_x, nozzle_y)
    pts_tip = np.array([
        [nozzle_x - 30, nozzle_y - 30],
        [nozzle_x + 30, nozzle_y - 30],
        [nozzle_x + 6, nozzle_y],
        [nozzle_x - 6, nozzle_y]
    ], np.int32)
    cv2.fillPoly(img, [pts_tip], brass_color)
    cv2.polylines(img, [pts_tip], True, brass_shadow, 2)
    
    # Small nozzle orifice (hole) at nozzle tip
    cv2.circle(img, (nozzle_x, nozzle_y), 3, (10, 10, 10), -1)

def generate_normal_sample(width=400, height=400):
    """Class 0: Normal extrusion and solid bed adhesion."""
    img = create_base_bed(width, height)
    nx, ny = 200, 140
    
    # Filament color (bright cyan/blue or vibrant red/white)
    filament_color = random.choice([(255, 180, 0), (0, 220, 255), (50, 50, 240), (220, 220, 220)])
    
    # Programmed line going from (200, 140) down onto bed to (200, 350)
    line_thickness = random.randint(10, 14) # Good squished first layer line width
    
    # Filament exiting nozzle tip directly onto bed surface
    cv2.line(img, (nx, ny + 2), (nx, 350), filament_color, line_thickness, cv2.LINE_AA)
    
    # Highlight shadow & texture on deposited filament
    cv2.line(img, (nx - line_thickness//4, ny + 2), (nx - line_thickness//4, 350), (255, 255, 255), 2, cv2.LINE_AA)
    
    draw_nozzle(img, nx, ny)
    return img

def generate_filament_not_deposited_sample(width=400, height=400, failure_type="no_extrusion"):
    """Class 1: Model 1 - Filament Not Deposited due to supply/extrusion failure."""
    img = create_base_bed(width, height)
    nx, ny = 200, 140
    
    filament_color = (0, 220, 255)
    
    if failure_type == "no_extrusion":
        # No filament comes out of nozzle at all. Empty print path on bed.
        # No line drawn on bed!
        pass
    elif failure_type == "insufficient_75":
        # 75% reduction: Extremely thin, starved line with broken segments
        cv2.line(img, (nx, ny + 2), (nx, 200), filament_color, 2, cv2.LINE_AA)
        cv2.line(img, (nx, 260), (nx, 300), filament_color, 2, cv2.LINE_AA)
    elif failure_type == "insufficient_50":
        # 50% reduction: Very thin line (under-extrusion)
        cv2.line(img, (nx, ny + 2), (nx, 350), filament_color, 4, cv2.LINE_AA)
    elif failure_type == "intermittent":
        # Intermittent extrusion: Gaps along path
        for y_start in range(ny + 10, 350, 40):
            cv2.line(img, (nx, y_start), (nx, y_start + 15), filament_color, 6, cv2.LINE_AA)
            
    draw_nozzle(img, nx, ny)
    return img

def generate_filament_not_adhering_sample(width=400, height=400, failure_type="floating_thread"):
    """Class 2: Model 2 - Filament Extruding but Not Adhering to Bed."""
    img = create_base_bed(width, height)
    nx, ny = 200, 140
    
    filament_color = (0, 220, 255)
    
    if failure_type == "floating_thread":
        # Filament exits nozzle but hangs in air as a thin wavy thread, not touching bed
        pts = []
        for t in range(0, 180, 5):
            x = int(nx + 35 * np.sin(t / 15.0))
            y = int(ny + 5 + t)
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, filament_color, 3, cv2.LINE_AA)
        
    elif failure_type == "dragging_loops":
        # Filament dragging behind nozzle in squiggles/loops above/on bed without sticking
        pts = []
        for t in range(0, 200, 5):
            x = int(nx + 50 * np.sin(t / 10.0))
            y = int(ny + 10 + t * 0.9)
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, filament_color, 4, cv2.LINE_AA)
        
    elif failure_type == "partially_lifted":
        # Attached near bottom, but middle is floating wavy in air
        cv2.line(img, (nx, 280), (nx, 350), filament_color, 10, cv2.LINE_AA) # Attached bottom
        pts = []
        for t in range(0, 140, 5):
            x = int(nx + 25 * np.sin(t / 12.0))
            y = int(ny + 5 + t)
            pts.append([x, y])
        pts = np.array(pts, np.int32)
        cv2.polylines(img, [pts], False, filament_color, 4, cv2.LINE_AA)
        
    draw_nozzle(img, nx, ny)
    return img

def generate_uncertain_sample(width=400, height=400):
    """Class 3: Uncertain / Other ambiguous images."""
    img = create_base_bed(width, height)
    # Heavy blur & brightness drop (e.g. camera out of focus or light blocked)
    img = cv2.GaussianBlur(img, (31, 31), 0)
    img = cv2.convertScaleAbs(img, alpha=0.4, beta=-30)
    return img

def generate_full_dataset(samples_per_category=100):
    """Generates synthetic dataset images for all categories and severities."""
    print("Generating synthetic 3D Printer First Layer Defect Dataset...")
    
    for cat in CATEGORIES:
        folder = os.path.join(DATASET_DIR, cat)
        os.makedirs(folder, exist_ok=True)
        
    count = 0
    
    # 1. Normal samples
    normal_dir = os.path.join(DATASET_DIR, "normal")
    for i in range(samples_per_category):
        img = generate_normal_sample()
        cv2.imwrite(os.path.join(normal_dir, f"normal_{i+1:04d}.png"), img)
        count += 1
        
    # 2. Filament Not Deposited (Model 1) samples across severities
    not_dep_dir = os.path.join(DATASET_DIR, "filament_not_deposited")
    failures = ["no_extrusion", "insufficient_75", "insufficient_50", "intermittent"]
    for i in range(samples_per_category):
        ftype = failures[i % len(failures)]
        img = generate_filament_not_deposited_sample(failure_type=ftype)
        cv2.imwrite(os.path.join(not_dep_dir, f"not_deposited_{ftype}_{i+1:04d}.png"), img)
        count += 1
        
    # 3. Filament Not Adhering (Model 2) samples across severities
    not_adh_dir = os.path.join(DATASET_DIR, "filament_not_adhering")
    adh_failures = ["floating_thread", "dragging_loops", "partially_lifted"]
    for i in range(samples_per_category):
        ftype = adh_failures[i % len(adh_failures)]
        img = generate_filament_not_adhering_sample(failure_type=ftype)
        cv2.imwrite(os.path.join(not_adh_dir, f"not_adhering_{ftype}_{i+1:04d}.png"), img)
        count += 1
        
    # 4. Uncertain samples
    uncertain_dir = os.path.join(DATASET_DIR, "uncertain")
    for i in range(30):
        img = generate_uncertain_sample()
        cv2.imwrite(os.path.join(uncertain_dir, f"uncertain_{i+1:04d}.png"), img)
        count += 1
        
    print(f"[SUCCESS] Dataset generation complete! Created {count} images in '{DATASET_DIR}/'.")

if __name__ == "__main__":
    generate_full_dataset(samples_per_category=100)
