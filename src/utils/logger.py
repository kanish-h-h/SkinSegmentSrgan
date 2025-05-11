import sqlite3
from pathlib import Path
import os


def debug_db_state():
    """Print current DB state for debugging."""
    db_path = "data/metadata.db"
    if not os.path.exists(db_path):
        print("[!] Database file does not exist!")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        print(f"[Debug] Existing tables: {tables}")

        # Check record counts
        if 'segmentation' in tables:
            cursor.execute("SELECT COUNT(*) FROM segmentation;")
            print(f"[Debug] Segmentation records: {cursor.fetchone()[0]}")

        if 'srgan' in tables:
            cursor.execute("SELECT COUNT(*) FROM srgan;")
            print(f"[Debug] SRGAN records: {cursor.fetchone()[0]}")

    finally:
        conn.close()


def init_db():
    """Ensure database and tables exist."""
    db_path = Path("data/metadata.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    try:
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS segmentation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            mask_path TEXT NOT NULL,
            split_type TEXT CHECK(split_type IN ('train', 'val', 'test')),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(image_path, mask_path)  -- Prevent duplicates
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS srgan (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lr_path TEXT NOT NULL,
            hr_path TEXT NOT NULL,
            split_type TEXT CHECK(split_type IN ('train', 'val', 'test')),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(lr_path, hr_path)  -- Prevent duplicates
        )
        """)

        conn.commit()
        print("[✓] Database initialized successfully")
    except sqlite3.Error as e:
        print(f"[!] DB Initialization Error: {e}")
    finally:
        conn.close()


def log_to_db(filepaths, split_type, table_name):
    """Bulk insert with duplicate handling."""
    init_db()
    conn = sqlite3.connect("data/metadata.db")
    cursor = conn.cursor()
    inserted_count = 0

    try:
        for path in filepaths:
            try:
                if table_name == "segmentation":
                    cursor.execute(
                        """INSERT OR IGNORE INTO segmentation
                        (image_path, mask_path, split_type)
                        VALUES (?, ?, ?)""",
                        (str(path["image"]), str(path["mask"]), split_type)
                    )
                elif table_name == "srgan":
                    cursor.execute(
                        """INSERT OR IGNORE INTO srgan
                        (lr_path, hr_path, split_type)
                        VALUES (?, ?, ?)""",
                        (str(path["lr"]), str(path["hr"]), split_type)
                    )
                inserted_count += cursor.rowcount
            except sqlite3.Error as e:
                print(f"[!] Error inserting {path}: {e}")

        conn.commit()
        print(f"[✓] Inserted {inserted_count} new records into {table_name}")
        print(f"[!] Skipped {len(filepaths) - inserted_count} duplicates")

    except Exception as e:
        print(f"[!] Transaction Error: {e}")
        conn.rollback()
    finally:
        conn.close()


def process_and_insert_all_data():
    """Fixed version that processes all splits correctly"""
    seg_base = Path("data/processed/segmentation")
    srgan_base = Path("data/processed/srgan")

    for split_type in ["train", "val", "test"]:
        # ===== Segmentation =====
        seg_img_dir = seg_base / split_type / "images"
        seg_mask_dir = seg_base / split_type / "masks"

        if not seg_img_dir.exists():
            print(f"[!] Skipping segmentation {split_type} - folder missing")
            continue

        seg_data = []
        for img_file in seg_img_dir.glob("*.*"):
            if img_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                mask_file = seg_mask_dir / img_file.name
                if mask_file.exists():
                    seg_data.append({
                        "image": f"{split_type}/images/{img_file.name}",
                        "mask": f"{split_type}/masks/{img_file.name}",
                        "split_type": split_type
                    })

        # ===== SRGAN =====
        srgan_hr_dir = srgan_base / split_type / "hr"
        srgan_lr_dir = srgan_base / split_type / "lr"

        if not srgan_hr_dir.exists():
            print(f"[!] Skipping SRGAN {split_type} - folder missing")
            continue

        srgan_data = []
        for hr_file in srgan_hr_dir.glob("*.*"):
            if hr_file.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                lr_file = srgan_lr_dir / f"{hr_file.stem}_x4{hr_file.suffix}"
                if lr_file.exists():
                    srgan_data.append({
                        "lr": f"{split_type}/lr/{lr_file.name}",
                        "hr": f"{split_type}/hr/{hr_file.name}",
                        "split_type": split_type
                    })

        # ===== Insert Data =====
        if seg_data:
            log_to_db(seg_data, split_type, "segmentation")
            print(
                f"Inserted {len(seg_data)} segmentation records for {split_type}")

        if srgan_data:
            log_to_db(srgan_data, split_type, "srgan")
            print(f"Inserted {len(srgan_data)} SRGAN records for {split_type}")


if __name__ == "__main__":
    process_and_insert_all_data()


if __name__ == "__main__":
    process_and_insert_all_data()
