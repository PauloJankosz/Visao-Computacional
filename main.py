import argparse

from texseg.dataset import load_dataset
from texseg.features import N_SCALES, WINDOW, dimension_names, extract_features
from texseg.filters import build_filter_bank


def main():
    parser = argparse.ArgumentParser(description="dataset, filtros e vetores de textura")
    parser.add_argument("--dataset", default="data/dataset")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    if len(dataset.images) == 0:
        return

    height, width = dataset.images[0].shape
    print(f"\n{len(dataset.images)} imagens carregadas de '{args.dataset}'")
    print(f"Tamanho: {width}x{height} pixels, tons de cinza")

    filters = build_filter_bank()
    print(f"\n{len(filters)} filtros x {N_SCALES} escalas = {len(filters) * N_SCALES} dimensoes")

    vectors = extract_features(dataset.images[0], filters)
    grid_size = height // WINDOW
    n_regions = grid_size * grid_size

    print(f"\nPrimeira imagem ({dataset.names[0]}):")
    print(f"  grade: {grid_size}x{grid_size} = {n_regions} regioes")
    print(f"  shape dos vetores: {vectors.shape}")

    names = dimension_names(filters)
    print(f"  exemplo regiao (0, 0):")
    for i in range(min(4, len(names))):
        print(f"    {names[i]} = {vectors[0, 0, i]:.4f}")
    print(f"    ...")


if __name__ == "__main__":
    main()
