import argparse

from texseg.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser(description="carrega o dataset de imagens")
    parser.add_argument("--dataset", default="data/dataset")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if len(dataset.images) == 0:
        return

    height, width = dataset.images[0].shape
    print(f"\n{len(dataset.images)} imagens carregadas de '{args.dataset}'")
    print(f"Tamanho: {width}x{height} pixels, tons de cinza")


if __name__ == "__main__":
    main()
