import sqlite3


def test_metadata_tables():
    """Verify SQLite tables exist with correct schema."""
    conn = sqlite3.connect("data/metadata.db")
    cursor = conn.cursor()

    # Segmentation table
    cursor.execute("PRAGMA table_info(segmentation)")
    seg_columns = cursor.fetchall()
    assert any(
        "image_path" in col for col in seg_columns), "Missing image_path column!"
    assert any(
        "split_type" in col for col in seg_columns), "Missing split_type column!"

    # SRGAN table
    cursor.execute("PRAGMA table_info(srgan)")
    srgan_columns = cursor.fetchall()
    assert any(
        "lr_path" in col for col in srgan_columns), "Missing lr_path column!"
    assert any(
        "hr_path" in col for col in srgan_columns), "Missing hr_path column!"

    conn.close()


def test_metadata_integrity():
    """Check that metadata entries match processed files."""
    conn = sqlite3.connect("data/metadata.db")
    cursor = conn.cursor()

    # Segmentation
    cursor.execute(
        "SELECT COUNT(*) FROM segmentation WHERE split_type='train'")
    db_train_count = cursor.fetchone()[0]
    actual_train_count = len(os.listdir(
        "data/processed/segmentation/train/images"))
    assert db_train_count == actual_train_count, "Metadata count mismatch for train split!"

    conn.close()
