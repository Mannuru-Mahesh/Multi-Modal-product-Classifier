"""
generate_data.py
------------------
Creates a SYNTHETIC multi-modal dataset for product classification:
  - An RGB image (64x64 PNG) for each product, saved to data/images/
  - A short text description for each product
  - A category label (one of 5 classes)

Why synthetic?
  - Works fully offline, no downloads, no licensing issues
  - We control how "hard" the problem is — neither image nor text
    alone is perfectly separable, but COMBINING them is. This is
    the whole point of a multi-modal fusion project: it should
    demonstrably outperform single-modality baselines.

Categories: Electronics, Clothing, Furniture, Toys, Books

Each category has:
  - A characteristic visual style (colors + shapes drawn with PIL)
  - A pool of vocabulary for generating text descriptions

We deliberately make each modality UNRELIABLE on a subset of
examples:
  - ~25% of images get a large "occlusion" block (simulating a
    corrupted / obstructed product photo) — the image carries
    almost no category signal for these samples.
  - ~25% of text descriptions are replaced with generic,
    category-agnostic boilerplate (simulating a low-effort
    listing) — the text carries almost no category signal.
These are INDEPENDENT, so ~6% of samples have BOTH degraded
(genuinely hard "no clear signal" cases) and ~56% have BOTH clean.
This setup directly tests whether a FUSION model can lean on
whichever modality is informative for a given sample — the central
promise of multi-modal learning.
"""

import os
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

CATEGORIES = ['Electronics', 'Clothing', 'Furniture', 'Toys', 'Books']
N_PER_CLASS = 200          # 200 images per category = 1000 total
IMG_SIZE = 64
OUT_DIR = 'data/images'
os.makedirs(OUT_DIR, exist_ok=True)


# =================================================================
# VISUAL STYLES — color palettes per category
# =================================================================
COLOR_PALETTES = {
    'Electronics': [(40, 40, 50), (60, 90, 140), (20, 20, 30), (100, 100, 120)],   # dark / metallic
    'Clothing':    [(230, 120, 150), (120, 180, 230), (240, 200, 80), (180, 120, 220)],  # bright/pastel
    'Furniture':   [(120, 80, 50), (160, 110, 70), (90, 60, 40), (200, 170, 130)],  # wood tones
    'Toys':        [(230, 50, 50), (250, 210, 30), (50, 130, 230), (60, 200, 90)],  # primary colors
    'Books':       [(230, 225, 210), (200, 60, 60), (50, 90, 150), (40, 160, 100)], # cream + cover colors
}

BG_COLORS = {
    'Electronics': (25, 25, 30),
    'Clothing':    (250, 248, 245),
    'Furniture':   (235, 225, 210),
    'Toys':        (245, 245, 245),
    'Books':       (245, 240, 230),
}


def draw_electronics(draw, palette):
    """Screen + buttons + grid lines."""
    draw.rectangle([8, 8, 56, 44], fill=palette[1], outline=palette[0], width=2)
    for i in range(4):
        x = 12 + i * 11
        draw.line([(x, 10), (x, 42)], fill=palette[2], width=1)
    for _ in range(3):
        cx, cy = random.randint(12, 52), random.randint(48, 58)
        draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=palette[3])


def draw_clothing(draw, palette):
    """Soft rounded shapes (shirt/dress silhouette)."""
    draw.ellipse([14, 6, 50, 40], fill=palette[1])           # body
    draw.polygon([(14, 12), (4, 22), (12, 26)], fill=palette[2])   # left sleeve
    draw.polygon([(50, 12), (60, 22), (52, 26)], fill=palette[2])  # right sleeve
    draw.ellipse([22, 40, 42, 60], fill=palette[3])          # bottom hem


def draw_furniture(draw, palette):
    """Table/chair silhouette."""
    draw.rectangle([6, 16, 58, 26], fill=palette[1], outline=palette[0], width=2)  # tabletop
    for x in (10, 50):
        draw.rectangle([x, 26, x+4, 58], fill=palette[2])  # legs
    draw.rectangle([14, 30, 50, 40], fill=palette[3])      # drawer/shelf


def draw_toys(draw, palette):
    """Star + circle + triangle — playful shapes."""
    draw.ellipse([6, 6, 28, 28], fill=palette[0])          # ball
    draw.polygon([(45, 8), (60, 28), (30, 28)], fill=palette[1])  # triangle block
    # simple 5-point star
    cx, cy, r = 44, 46, 14
    pts = []
    for i in range(10):
        ang = np.pi/2 + i * np.pi/5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rad*np.cos(ang), cy - rad*np.sin(ang)))
    draw.polygon(pts, fill=palette[2])


def draw_books(draw, palette):
    """Stacked book spines with horizontal 'text' lines."""
    y = 6
    heights = [16, 14, 18]
    for i, h in enumerate(heights):
        color = palette[(i + 1) % len(palette)]
        draw.rectangle([8, y, 56, y + h], fill=color, outline=(60, 60, 60), width=1)
        for ly in range(y + 4, y + h - 2, 4):
            draw.line([(12, ly), (50, ly)], fill=(255, 255, 255), width=1)
        y += h + 3


DRAW_FUNCS = {
    'Electronics': draw_electronics,
    'Clothing':    draw_clothing,
    'Furniture':   draw_furniture,
    'Toys':        draw_toys,
    'Books':       draw_books,
}


def add_noise(img_array, level=10):
    """Add light Gaussian pixel noise so images aren't perfectly clean."""
    noise = np.random.normal(0, level, img_array.shape)
    noisy = np.clip(img_array.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def occlude_image(img_array):
    """
    Simulate a completely corrupted / failed-to-load product photo
    by replacing the ENTIRE image with random static. This destroys
    100% of the shape AND color signal — a true "this modality is
    missing" scenario (e.g., broken image link, camera error,
    placeholder thumbnail).
    """
    h, w = img_array.shape[:2]
    static = np.random.randint(90, 190, size=(h, w, 3), dtype=np.uint8)
    return static


def generate_image(category, occluded=False):
    """Draw a synthetic product image for `category`."""
    palette = list(COLOR_PALETTES[category])
    random.shuffle(palette)
    bg = BG_COLORS[category]

    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), bg)
    draw = ImageDraw.Draw(img)
    DRAW_FUNCS[category](draw, palette)

    arr = np.array(img)
    arr = add_noise(arr, level=10)
    if occluded:
        arr = occlude_image(arr)
    return Image.fromarray(arr)


# =================================================================
# TEXT VOCABULARY — phrases per category
# =================================================================
VOCAB = {
    'Electronics': [
        "wireless", "bluetooth", "rechargeable battery", "HD display", "fast charging",
        "noise cancelling", "smart device", "touchscreen", "high performance processor",
        "USB-C port", "compact design", "long battery life"
    ],
    'Clothing': [
        "soft cotton fabric", "comfortable fit", "machine washable", "stylish design",
        "available in multiple colors", "breathable material", "perfect for everyday wear",
        "trendy pattern", "lightweight fabric", "true to size", "casual style", "premium stitching"
    ],
    'Furniture': [
        "solid wood construction", "easy to assemble", "spacious storage", "modern design",
        "durable finish", "fits any room", "comfortable seating", "scratch resistant surface",
        "elegant look", "sturdy frame", "compact footprint", "ideal for living room"
    ],
    'Toys': [
        "fun for all ages", "encourages creativity", "bright colors", "safe materials",
        "great gift idea", "easy to clean", "durable plastic", "interactive play",
        "educational toy", "perfect for kids", "lightweight and portable", "battery free"
    ],
    'Books': [
        "bestselling novel", "gripping storyline", "award winning author", "paperback edition",
        "hardcover collection", "perfect for book lovers", "great gift idea", "easy to read",
        "illustrated pages", "thought provoking", "classic literature", "page turner"
    ],
}

TEMPLATES = [
    "This product features {a} and {b}.",
    "A great choice if you want {a} with {b}.",
    "Customers love the {a} and {b}.",
    "Designed with {a}, also offering {b}.",
    "Comes with {a} and is known for {b}.",
]

# Generic / category-agnostic descriptions — simulate a listing with
# a low-effort, boilerplate description that gives NO category signal.
GENERIC_TEMPLATES = [
    "Great value for the price, highly recommended by customers.",
    "Fast shipping and excellent packaging, very satisfied overall.",
    "Good quality product, works as described, would buy again.",
    "Popular item with thousands of positive reviews this month.",
    "Ships within 24 hours, hassle-free returns, top rated seller.",
    "Limited time offer, available now with free delivery.",
]


def generate_text(category, generic=False):
    """
    Generate a description for `category` using its vocabulary.
    If `generic=True`, return a category-agnostic boilerplate
    description instead (simulates a missing/uninformative text
    signal — the text-only model gets no useful clue here).
    """
    if generic:
        return random.choice(GENERIC_TEMPLATES)

    pool = VOCAB[category]
    a, b = random.sample(pool, 2)
    template = random.choice(TEMPLATES)
    return template.format(a=a, b=b)


# =================================================================
# MAIN GENERATION LOOP
# =================================================================
records = []
NOISE_RATE = 0.25  # 25% of samples get an occluded image / generic text

for category in CATEGORIES:
    for i in range(N_PER_CLASS):
        image_occluded = random.random() < NOISE_RATE
        text_generic   = random.random() < NOISE_RATE

        img = generate_image(category, occluded=image_occluded)
        text = generate_text(category, generic=text_generic)

        filename = f"{category}_{i:04d}.png"
        img.save(os.path.join(OUT_DIR, filename))

        records.append({
            'image_path': f"images/{filename}",
            'description': text,
            'category': category,
            'image_occluded': image_occluded,
            'text_generic': text_generic,
        })

df = pd.DataFrame(records)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
df.to_csv('data/metadata.csv', index=False)

print(f"✅ Generated {len(df)} samples across {len(CATEGORIES)} categories")
print(f"   Images saved to: data/images/")
print(f"   Metadata saved to: data/metadata.csv")
print(f"\nClass distribution:\n{df['category'].value_counts()}")
print(f"\nSamples with occluded images: {df['image_occluded'].sum()} ({df['image_occluded'].mean()*100:.0f}%)")
print(f"Samples with generic text:    {df['text_generic'].sum()} ({df['text_generic'].mean()*100:.0f}%)")
print(f"Samples with BOTH (hard cases): {(df['image_occluded'] & df['text_generic']).sum()}")
print(f"\nSample rows:")
print(df.head(3).to_string(index=False))
