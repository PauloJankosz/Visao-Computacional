from __future__ import annotations
import logging
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Caminhos
DATASET_PATH = Path(__file__).resolve().parent / "iris.data"
OUTPUT_DIR = Path(__file__).resolve().parent / "imagens"

FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
COLUMN_NAMES = [*FEATURE_COLUMNS, TARGET_COLUMN]

FEATURE_LABELS = {
    "sepal_length": "Comprimento Sépala (cm)",
    "sepal_width": "Largura Sépala (cm)",
    "petal_length": "Comprimento Pétala (cm)",
    "petal_width": "Largura Pétala (cm)",
}
FEATURE_LABELS_SHORT = ["Compr. Sép.", "Larg. Sép.", "Compr. Pét.", "Larg. Pét."]

CLASS_COLORS = {
    "Iris-setosa": "#e74c3c",
    "Iris-versicolor": "#2ecc71",
    "Iris-virginica": "#3498db",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
K_RANGE = range(1, 21)
FIGURE_DPI = 150


def _save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=FIGURE_DPI)
    plt.close(fig)


def _section(title):
    sep = "=" * 70
    log.info("\n%s\n  %s\n%s", sep, title, sep)

def load_iris(path=DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMN_NAMES)
    return df[df[TARGET_COLUMN].str.strip() != ""]

def _stats_of(series: pd.Series):
    return series.mean(), series.std(), series.mode().iloc[0]


def print_overall_stats(df):
    _section("ESTATÍSTICAS DESCRITIVAS GERAIS")
    for feat in FEATURE_COLUMNS:
        mean, std, mode = _stats_of(df[feat])
        log.info("\n  %s:", FEATURE_LABELS[feat])
        log.info("    Média:  %.4f", mean)
        log.info("    Desvio: %.4f", std)
        log.info("    Moda:   %.2f", mode)


def print_stats_by_class(df):
    _section("ESTATÍSTICAS POR CATEGORIA")
    for species, group in df.groupby(TARGET_COLUMN):
        log.info("\n  >>> %s (%d amostras)", species, len(group))
        for feat in FEATURE_COLUMNS:
            mean, std, mode = _stats_of(group[feat])
            log.info(
                "    %s:  média=%.4f  desvio=%.4f  moda=%.2f",
                FEATURE_LABELS[feat], mean, std, mode,
            )


def print_class_frequency(df):
    _section("FREQUÊNCIA DAS CATEGORIAS")
    freq = df[TARGET_COLUMN].value_counts()
    for species, n in freq.items():
        log.info("    %s: %d (%.1f%%)", species, n, n / len(df) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    freq.plot(kind="bar", color=[CLASS_COLORS[c] for c in freq.index], ax=ax, edgecolor="black")
    ax.set(title="Frequência das Categorias", ylabel="Quantidade", xlabel="Classe")
    plt.xticks(rotation=0)
    _save_fig(fig, "01_frequencia_categorias.png")


# Scatter + correlação 

def _scatter_by_class(df, x_col, y_col, title, filename):
    fig, ax = plt.subplots(figsize=(10, 7))
    for species, color in CLASS_COLORS.items():
        mask = df[TARGET_COLUMN] == species
        ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                   c=color, label=species, alpha=0.7, edgecolors="k", s=60)
    ax.set_xlabel(FEATURE_LABELS[x_col], fontsize=13)
    ax.set_ylabel(FEATURE_LABELS[y_col], fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    _save_fig(fig, filename)


def _print_pearson(df, x_col, y_col, header):
    _section(header)
    for species, group in df.groupby(TARGET_COLUMN):
        log.info("    %s: r = %.4f", species, group[x_col].corr(group[y_col]))
    log.info("    Geral:           r = %.4f", df[x_col].corr(df[y_col]))


def analyze_sepal(df):
    _scatter_by_class(df, "sepal_length", "sepal_width",
                      "Relação Comprimento × Largura da Sépala por Categoria",
                      "02_sepala_comprimento_largura.png")
    _print_pearson(df, "sepal_length", "sepal_width",
                   "CORRELAÇÃO: COMPRIMENTO × LARGURA DA SÉPALA")


def analyze_petal(df):
    _scatter_by_class(df, "petal_length", "petal_width",
                      "Relação Comprimento × Largura da Pétala por Categoria",
                      "03_petala_comprimento_largura.png")
    _print_pearson(df, "petal_length", "petal_width",
                   "CORRELAÇÃO: COMPRIMENTO × LARGURA DA PÉTALA")

def plot_distributions(df):
    # Histogramas
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for feat, ax in zip(FEATURE_COLUMNS, axes.flatten()):
        for species, color in CLASS_COLORS.items():
            vals = df.loc[df[TARGET_COLUMN] == species, feat]
            ax.hist(vals, bins=15, alpha=0.5, color=color, label=species, edgecolor="black")
        ax.set_title(FEATURE_LABELS[feat], fontsize=13, fontweight="bold")
        ax.set(xlabel="Valor (cm)", ylabel="Frequência")
        ax.legend(fontsize=9)
    fig.suptitle("Distribuição das Categorias nas 4 Dimensões",
                 fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, "04_distribuicao_4d.png")

    # Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for feat, ax in zip(FEATURE_COLUMNS, axes.flatten()):
        df.boxplot(column=feat, by=TARGET_COLUMN, ax=ax, patch_artist=True,
                   boxprops={"facecolor": "lightblue"},
                   medianprops={"color": "red", "linewidth": 2})
        ax.set_title(FEATURE_LABELS[feat], fontsize=13, fontweight="bold")
        ax.set(xlabel="Classe", ylabel="Valor (cm)")
    fig.suptitle("Boxplots por Categoria e Variável",
                 fontsize=15, fontweight="bold", y=1.01)
    _save_fig(fig, "05_boxplots.png")

    # Pairplot
    g = sns.pairplot(df, hue=TARGET_COLUMN, palette=CLASS_COLORS,
                     diag_kind="kde", plot_kws={"alpha": 0.6, "edgecolor": "k", "s": 40})
    g.figure.suptitle("Pairplot — Todas as Variáveis por Categoria",
                      y=1.02, fontsize=15, fontweight="bold")
    _save_fig(g.figure, "06_pairplot.png")


# PCA

def run_pca(features, labels) -> tuple[PCA, np.ndarray]:
    _section("PRINCIPAL COMPONENT ANALYSIS (PCA)")

    pca = PCA()
    projected = pca.fit_transform(features)

    # Variância explicada
    log.info("\n  Variância explicada por componente:")
    for i, (v, r) in enumerate(zip(pca.explained_variance_, pca.explained_variance_ratio_)):
        log.info("    PC%d: variância = %.4f  (%.2f%%)", i + 1, v, r * 100)
    cum2 = pca.explained_variance_ratio_[:2].sum() * 100
    log.info("    Acumulada (PC1+PC2): %.2f%%", cum2)

    # Loadings
    log.info("\n  Loadings (contribuição das variáveis):")
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(FEATURE_COLUMNS))],
        index=[FEATURE_LABELS[f] for f in FEATURE_COLUMNS],
    )
    log.info(loadings_df.to_string())

    _plot_pca_variance(pca)
    _plot_pca_2d(pca, projected, labels)
    _plot_biplot(pca, projected, labels)

    return pca, projected


def _plot_pca_variance(pca):
    n = len(pca.explained_variance_ratio_)
    ticks = range(1, n + 1)
    pct = pca.explained_variance_ratio_ * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(ticks, pct, color="steelblue", edgecolor="black")
    ax1.set(xlabel="Componente Principal", ylabel="Variância Explicada (%)")
    ax1.set_title("Variância Explicada por Componente", fontweight="bold")
    ax1.set_xticks(list(ticks))

    ax2.plot(ticks, np.cumsum(pct), "bo-", linewidth=2, markersize=8)
    ax2.axhline(y=95, color="r", linestyle="--", label="95%")
    ax2.set(xlabel="Número de Componentes", ylabel="Variância Acumulada (%)")
    ax2.set_title("Variância Acumulada (Scree Plot)", fontweight="bold")
    ax2.set_xticks(list(ticks))
    ax2.legend()

    _save_fig(fig, "07_pca_variancia.png")


def _plot_pca_2d(pca, projected, labels):
    ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 7))
    for species, color in CLASS_COLORS.items():
        m = labels == species
        ax.scatter(projected[m, 0], projected[m, 1],
                   c=color, label=species, alpha=0.7, edgecolors="k", s=60)
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)", fontsize=13)
    ax.set_title("Projeção PCA (2 Componentes Principais)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    _save_fig(fig, "08_pca_projecao_2d.png")


def _plot_biplot(pca, projected, labels):
    ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(10, 8))
    for species, color in CLASS_COLORS.items():
        m = labels == species
        ax.scatter(projected[m, 0], projected[m, 1], c=color, label=species, alpha=0.5, s=40)

    # Vetores das variáveis originais
    coefs = pca.components_[:2, :].T
    scale = 3.0
    for i, label in enumerate(FEATURE_LABELS_SHORT):
        dx, dy = coefs[i, 0] * scale, coefs[i, 1] * scale
        ax.arrow(0, 0, dx, dy, head_width=0.08, head_length=0.05, fc="black", ec="black")
        ax.text(dx * 1.15, dy * 1.15, label, fontsize=11, fontweight="bold", ha="center")

    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}%)", fontsize=13)
    ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}%)", fontsize=13)
    ax.set_title("Biplot PCA — Variáveis e Amostras", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    _save_fig(fig, "09_pca_biplot.png")


# KNN

def _find_best_k(x_train, x_test, y_train, y_test):
    """Varre K_RANGE e retorna (melhor_k, melhor_acurácia, lista_acurácias)."""
    accs = []
    for k in K_RANGE:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        accs.append(knn.score(x_test, y_test))
    idx = int(np.argmax(accs))
    return list(K_RANGE)[idx], accs[idx], accs


def run_knn(features, labels, classes: Sequence[str]):
    _section("K-NN: CLASSIFICAÇÃO")

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels,
    )
    log.info("\n  Amostras de treino: %d", len(x_train))
    log.info("  Amostras de teste:  %d", len(x_test))

    best_k, best_acc, accs = _find_best_k(x_train, x_test, y_train, y_test)
    log.info("\n  Melhor K: %d (acurácia: %.2f%%)", best_k, best_acc * 100)

    # Gráfico K vs acurácia
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K_RANGE), [a * 100 for a in accs], "bo-", linewidth=2, markersize=6)
    ax.axvline(x=best_k, color="red", linestyle="--", label=f"Melhor K={best_k}")
    ax.set(xlabel="K (número de vizinhos)", ylabel="Acurácia (%)")
    ax.set_title("Acurácia do K-NN para diferentes valores de K", fontsize=14, fontweight="bold")
    ax.set_xticks(list(K_RANGE))
    ax.legend(fontsize=12)
    _save_fig(fig, "10_knn_acuracia_k.png")

    # Modelo final e métricas
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)

    log.info("\n  Relatório de Classificação (K=%d):", best_k)
    log.info(classification_report(y_test, y_pred, target_names=classes))

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set(xlabel="Predito", ylabel="Real")
    ax.set_title(f"Matriz de Confusão — K-NN (K={best_k})", fontsize=14, fontweight="bold")
    _save_fig(fig, "11_knn_matriz_confusao.png")

    # Fronteira de decisão projetada em 2D via PCA
    pca2 = PCA(n_components=2)
    train_2d = pca2.fit_transform(x_train)
    test_2d = pca2.transform(x_test)

    knn2d = KNeighborsClassifier(n_neighbors=best_k)
    knn2d.fit(train_2d, y_train)

    h = 0.05
    xx, yy = np.meshgrid(
        np.arange(train_2d[:, 0].min() - 1, train_2d[:, 0].max() + 1, h),
        np.arange(train_2d[:, 1].min() - 1, train_2d[:, 1].max() + 1, h),
    )
    class_list = list(classes)
    z = knn2d.predict(np.c_[xx.ravel(), yy.ravel()])
    z_num = np.array([class_list.index(c) for c in z]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.contourf(xx, yy, z_num, alpha=0.3, cmap="RdYlBu")
    for species, color in CLASS_COLORS.items():
        mt = y_test == species
        ax.scatter(test_2d[mt, 0], test_2d[mt, 1], c=color,
                   label=f"{species} (teste)", edgecolors="k", s=80)
    for species, color in CLASS_COLORS.items():
        mr = y_train == species
        ax.scatter(train_2d[mr, 0], train_2d[mr, 1], c=color, alpha=0.3, s=30, marker="x")
    ax.set(xlabel="PC1", ylabel="PC2")
    ax.set_title(f"Fronteira de Decisão K-NN (K={best_k}) no espaço PCA",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    _save_fig(fig, "12_knn_fronteira_decisao.png")

    return best_k, best_acc


# -- Main --

def main():
    plt.rcParams.update({"figure.figsize": (10, 7), "font.size": 12})
    sns.set_style("whitegrid")

    df = load_iris()
    log.info("=" * 70)
    log.info("       ANÁLISE DO DATASET IRIS — TRABALHO T4")
    log.info("=" * 70)
    log.info("\nTotal de amostras: %d", len(df))
    log.info("Variáveis: %d", len(FEATURE_COLUMNS))
    log.info("Classes: %s", df[TARGET_COLUMN].unique().tolist())

    print_overall_stats(df)
    print_stats_by_class(df)
    print_class_frequency(df)

    analyze_sepal(df)
    analyze_petal(df)
    plot_distributions(df)

    # Normaliza antes de PCA e K-NN
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURE_COLUMNS].values)
    y = df[TARGET_COLUMN].values
    classes = df[TARGET_COLUMN].unique().tolist()

    pca, _ = run_pca(X, y)
    best_k, best_acc = run_knn(X, y, classes)

    # Resumo
    _section("RESUMO FINAL")
    cum2 = pca.explained_variance_ratio_[:2].sum() * 100
    log.info(
        "\n"
        "  - Dataset com 150 amostras e 3 classes balanceadas (50 cada)\n"
        "  - 4 variáveis: comprimento/largura de sépala e pétala\n"
        "  - Iris-setosa é claramente separável das outras duas\n"
        "  - Iris-versicolor e Iris-virginica possuem sobreposição\n"
        "  - PCA com 2 componentes explica %.1f%% da variância\n"
        "  - K-NN com K=%d obteve acurácia de %.1f%% no teste\n"
        "  - As variáveis de pétala têm maior poder discriminativo\n"
        "\n  Gráficos salvos (01 a 12) em: %s",
        cum2, best_k, best_acc * 100, OUTPUT_DIR,
    )
    log.info("\nAnálise concluída com sucesso!")


if __name__ == "__main__":
    main()
