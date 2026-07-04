from src.phase1_config import DATA_DIR, PLOTS_DIR, SPLITS_DIR, IMAGE_SIZE, BATCH_SIZE
from src.phase1_dataset import scan_image_folder, get_class_distribution, split_dataset, save_splits
from src.phase1_eda import plot_class_distribution, plot_sample_grid
from src.phase1_preprocessing import build_dataloaders


def main():
    df = scan_image_folder(DATA_DIR)

    distribution = get_class_distribution(df)
    print("\nClass distribution:")
    print(distribution.to_string(index=False))

    plot_class_distribution(df, PLOTS_DIR / "class_distribution.png")
    plot_sample_grid(df, PLOTS_DIR / "sample_grid.png")

    train_df, val_df, test_df = split_dataset(df)
    save_splits(train_df, val_df, test_df, SPLITS_DIR)

    train_loader, val_loader, test_loader, label_to_index = build_dataloaders(
        train_df,
        val_df,
        test_df,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )

    images, labels = next(iter(train_loader))

    print("\nDataset size:", len(df))
    print("Train size:", len(train_df))
    print("Validation size:", len(val_df))
    print("Test size:", len(test_df))
    print("Batch image shape:", tuple(images.shape))
    print("Batch label shape:", tuple(labels.shape))
    print("Labels:", label_to_index)
    print("\nSaved files:")
    print(f"- {PLOTS_DIR / 'class_distribution.png'}")
    print(f"- {PLOTS_DIR / 'sample_grid.png'}")
    print(f"- {SPLITS_DIR / 'train.csv'}")
    print(f"- {SPLITS_DIR / 'val.csv'}")
    print(f"- {SPLITS_DIR / 'test.csv'}")


if __name__ == "__main__":
    main()
