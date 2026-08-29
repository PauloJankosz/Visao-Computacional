import argparse

from texseg.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="urban texture segmentation")
    parser.add_argument("--dataset", default="data/dataset")
    parser.add_argument("--output", default="results")
    parser.add_argument("--k", type=int, default=6, help="texture clusters per region")
    parser.add_argument("--k-images", type=int, default=4, dest="k_images", help="image classes")
    args = parser.parse_args()

    run(
        dataset_dir=args.dataset,
        out_dir=args.output,
        k=args.k,
        k_images=args.k_images,
    )


if __name__ == "__main__":
    main()
