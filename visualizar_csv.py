"""Genera un panel visual y reportes de calidad para un archivo CSV.

Requiere pandas y matplotlib: python -m pip install pandas matplotlib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


CSV_NAME = "ia_project_golden_clean.csv"
COLORS = {
    "blue": "#2563EB",
    "teal": "#0F766E",
    "orange": "#EA580C",
    "red": "#DC2626",
    "purple": "#7C3AED",
    "gray": "#64748B",
    "light": "#E2E8F0",
}


def find_default_csv() -> Path | None:
    """Busca el CSV junto al script, en la carpeta actual o en Visualizador."""
    candidates = [
        Path(__file__).resolve().parent / CSV_NAME,
        Path.cwd() / CSV_NAME,
    ]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / "Visualizador" / CSV_NAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def normalized_consent(value: object) -> str:
    text = str(value).strip().casefold()
    if text in {"sí", "si", "yes", "y", "true", "1"}:
        return "Sí"
    if text in {"no", "n", "false", "0"}:
        return "No"
    return "Vacío / otro"


def date_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    return pd.to_datetime(df[name], errors="coerce")


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["light"])
    ax.tick_params(axis="y", length=0, colors="#334155")
    ax.tick_params(axis="x", colors="#64748B")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)


def add_empty_message(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["gray"],
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def create_dashboard(
    df: pd.DataFrame,
    missing: pd.DataFrame,
    missing_cells: int,
    rows_with_missing: int,
    duplicate_ids: int,
    chronology_issues: int,
    source_name: str,
    output_path: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    })

    total_rows, total_columns = df.shape
    duplicate_id_label = "record_id" if "record_id" in df.columns else None
    duplicate_id_count = duplicate_ids if duplicate_id_label else 0
    cards = [
        ("REGISTROS", f"{total_rows:,}", COLORS["blue"]),
        ("COLUMNAS", f"{total_columns:,}", COLORS["teal"]),
        ("CELDAS VACÍAS", f"{missing_cells:,}", COLORS["orange"]),
        ("FILAS CON ALGÚN VACÍO", f"{rows_with_missing:,}", COLORS["purple"]),
        ("ID REPETIDOS", f"{duplicate_id_count:,}", COLORS["red"]),
    ]

    fig = plt.figure(figsize=(17, 12), facecolor="#F8FAFC")
    fig.suptitle("Exploración y calidad de datos", x=0.06, y=0.975,
                 ha="left", fontsize=22, fontweight="bold", color="#0F172A")
    fig.text(
        0.06, 0.941,
        f"{source_name}  |  Fechas de último acceso anteriores al registro: "
        f"{chronology_issues:,}",
        ha="left", fontsize=10, color=COLORS["gray"],
    )

    for index, (label, value, color) in enumerate(cards):
        x = 0.06 + index * 0.185
        card = fig.add_axes([x, 0.855, 0.17, 0.065])
        card.set_facecolor("#FFFFFF")
        for spine in card.spines.values():
            spine.set_color("#E2E8F0")
        card.set_xticks([])
        card.set_yticks([])
        card.text(0.06, 0.68, label, transform=card.transAxes,
                  fontsize=8, color=COLORS["gray"], fontweight="bold")
        card.text(0.06, 0.18, value, transform=card.transAxes,
                  fontsize=18, color=color, fontweight="bold")

    grid = fig.add_gridspec(
        3, 2, left=0.07, right=0.96, bottom=0.07, top=0.81,
        hspace=0.48, wspace=0.30,
    )

    # Valores faltantes por columna
    ax = fig.add_subplot(grid[0, 0])
    style_axis(ax)
    missing_by_column = missing.sum().sort_values(ascending=True)
    missing_by_column = missing_by_column[missing_by_column > 0].tail(12)
    if missing_by_column.empty:
        add_empty_message(ax, "No se detectaron valores vacíos")
    else:
        percentages = missing_by_column / max(total_rows, 1) * 100
        ax.barh(percentages.index, percentages.values, color=COLORS["orange"])
        ax.set_title("Campos incompletos (porcentaje de registros)", loc="left", pad=10)
        ax.set_xlabel("% de registros")
        ax.set_xlim(0, max(5, percentages.max() * 1.18))
        for y, (count, percentage) in enumerate(zip(missing_by_column, percentages)):
            ax.text(percentage + 0.15, y, f"{int(count):,}  ({percentage:.1f}%)",
                    va="center", fontsize=8, color="#334155")

    # Registros por mes
    ax = fig.add_subplot(grid[0, 1])
    style_axis(ax)
    signup = date_column(df, "signup_date")
    if signup.notna().any():
        monthly = signup.dropna().dt.to_period("M").value_counts().sort_index()
        monthly.index = monthly.index.to_timestamp()
        ax.plot(monthly.index, monthly.values, color=COLORS["blue"],
                linewidth=2, marker="o", markersize=3)
        ax.fill_between(monthly.index, monthly.values, color=COLORS["blue"], alpha=0.10)
        ax.set_title("Altas de clientes por mes", loc="left", pad=10)
        ax.set_ylabel("Registros")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.set_title("Altas de clientes por mes", loc="left", pad=10)
        add_empty_message(ax, "No hay fechas de alta válidas")

    # Distribución del gasto
    ax = fig.add_subplot(grid[1, 0])
    style_axis(ax)
    if "amount_spent_mxn" in df.columns:
        spent = pd.to_numeric(df["amount_spent_mxn"], errors="coerce").dropna()
    else:
        spent = pd.Series(dtype=float)
    if not spent.empty:
        ax.hist(spent, bins=30, color=COLORS["teal"], edgecolor="white")
        median = spent.median()
        ax.axvline(median, color=COLORS["red"], linestyle="--", linewidth=1.8,
                   label=f"Mediana: ${median:,.2f}")
        ax.set_title("Distribución del gasto", loc="left", pad=10)
        ax.set_xlabel("Monto (MXN)")
        ax.set_ylabel("Registros")
        ax.legend(frameon=False)
    else:
        ax.set_title("Distribución del gasto", loc="left", pad=10)
        add_empty_message(ax, "No hay valores numéricos para graficar")

    # Distribución por país
    ax = fig.add_subplot(grid[1, 1])
    style_axis(ax)
    if "country" in df.columns:
        countries = df["country"].fillna("Vacío").astype(str).value_counts().head(10).sort_values()
    else:
        countries = pd.Series(dtype=int)
    if not countries.empty:
        ax.barh(countries.index, countries.values, color=COLORS["purple"])
        ax.set_title("Registros por país", loc="left", pad=10)
        ax.set_xlabel("Registros")
        for y, value in enumerate(countries.values):
            ax.text(value + max(countries.max() * 0.01, 0.1), y, f"{value:,}",
                    va="center", fontsize=8, color="#334155")
    else:
        ax.set_title("Registros por país", loc="left", pad=10)
        add_empty_message(ax, "No existe una columna country")

    # Distribuciones de estado y plan
    ax = fig.add_subplot(grid[2, 0])
    style_axis(ax)
    category_rows: list[tuple[str, int, str]] = []
    for column, label, color in (
        ("customer_status", "Estado", COLORS["blue"]),
        ("plan", "Plan", COLORS["teal"]),
    ):
        if column in df.columns:
            counts = df[column].fillna("Vacío").astype(str).value_counts()
            category_rows.extend((f"{label}: {name}", int(count), color)
                                 for name, count in counts.items())
    if category_rows:
        labels = [row[0] for row in category_rows]
        values = [row[1] for row in category_rows]
        colors = [row[2] for row in category_rows]
        positions = list(range(len(labels)))
        ax.barh(positions, values, color=colors)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title("Clientes por estado y plan", loc="left", pad=10)
        ax.set_xlabel("Registros")
        for y, value in enumerate(values):
            ax.text(value + max(max(values) * 0.01, 0.1), y, f"{value:,}",
                    va="center", fontsize=8, color="#334155")
    else:
        ax.set_title("Clientes por estado y plan", loc="left", pad=10)
        add_empty_message(ax, "No se encontraron columnas de estado o plan")

    # Normaliza variantes comunes de consentimiento
    ax = fig.add_subplot(grid[2, 1])
    style_axis(ax)
    if "marketing_consent" in df.columns:
        consent = df["marketing_consent"].map(normalized_consent).value_counts()
        consent = consent.reindex(["Sí", "No", "Vacío / otro"], fill_value=0)
        consent_colors = [COLORS["teal"], COLORS["orange"], COLORS["gray"]]
        ax.bar(consent.index, consent.values, color=consent_colors)
        ax.set_title("Consentimiento de marketing (valores normalizados)", loc="left", pad=10)
        ax.set_ylabel("Registros")
        for x, value in enumerate(consent.values):
            ax.text(x, value + max(consent.max() * 0.015, 0.1), f"{value:,}",
                    ha="center", fontsize=9, color="#334155")
        raw_categories = df["marketing_consent"].nunique(dropna=True)
        ax.text(0.99, 0.93, f"{raw_categories} variantes originales",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color=COLORS["gray"])
    else:
        ax.set_title("Consentimiento de marketing", loc="left", pad=10)
        add_empty_message(ax, "No existe una columna marketing_consent")

    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera gráficas y un reporte de calidad para un CSV."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=find_default_csv(),
        help=f"Ruta del CSV (por defecto se busca {CSV_NAME} junto al script o en Visualizador).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "visualizaciones_csv",
        help="Carpeta donde guardar las gráficas y reportes.",
    )
    args = parser.parse_args()

    if args.csv is None:
        parser.error(
            f"No encontré {CSV_NAME}. Indica su ruta, por ejemplo: "
            'python visualizar_csv.py "C:\\ruta\\archivo.csv"'
        )
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        parser.error(f"No existe el archivo CSV: {csv_path}")

    try:
        df = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            dtype={"record_id": "string", "postal_code": "string"},
        )
    except (UnicodeDecodeError, pd.errors.ParserError, OSError) as error:
        parser.error(f"No se pudo leer el CSV {csv_path}: {error}")

    if df.empty:
        parser.error("El CSV no contiene registros.")

    blank_strings = df.apply(
        lambda column: column.astype("string").str.strip().eq("").fillna(False)
    )
    missing = df.isna() | blank_strings
    missing_cells = int(missing.to_numpy().sum())
    rows_with_missing = int(missing.any(axis=1).sum())

    duplicate_rows = int(df.duplicated().sum())
    if "record_id" in df.columns:
        duplicate_ids = int(df["record_id"].duplicated().sum())
        duplicate_id_values = int(
            df.loc[df["record_id"].duplicated(keep=False), "record_id"].nunique()
        )
    else:
        duplicate_ids = 0
        duplicate_id_values = 0

    signup = date_column(df, "signup_date")
    last_login = date_column(df, "last_login")
    chronology_issues = int(
        (signup.notna() & last_login.notna() & (last_login < signup)).sum()
    )

    if "email_valid" in df.columns:
        email_valid = df["email_valid"].astype("string").str.strip().str.casefold()
        invalid_email_flags = int(email_valid.isin({"false", "0", "no", "n"}).sum())
    else:
        invalid_email_flags = 0

    total_cells = int(df.shape[0] * df.shape[1])
    column_report = pd.DataFrame({
        "columna": df.columns,
        "registros_vacios": [int(missing[column].sum()) for column in df.columns],
        "porcentaje_vacio": [
            float(missing[column].mean() * 100) for column in df.columns
        ],
        "valores_unicos": [int(df[column].nunique(dropna=True)) for column in df.columns],
    }).sort_values("registros_vacios", ascending=False)

    metrics = [
        ("Registros", len(df)),
        ("Columnas", len(df.columns)),
        ("Celdas vacías", missing_cells),
        ("Porcentaje de celdas vacías", round(missing_cells / max(total_cells, 1) * 100, 2)),
        ("Filas con al menos un valor vacío", rows_with_missing),
        ("Filas duplicadas completas (ocurrencias extra)", duplicate_rows),
        ("Registros con record_id repetido (ocurrencias extra)", duplicate_ids),
        ("Valores record_id que aparecen más de una vez", duplicate_id_values),
        ("email_valid marcado como falso", invalid_email_flags),
        ("last_login anterior a signup_date", chronology_issues),
    ]
    if "marketing_consent" in df.columns:
        metrics.append((
            "Variantes originales de marketing_consent",
            int(df["marketing_consent"].nunique(dropna=True)),
        ))

    args.output.mkdir(parents=True, exist_ok=True)
    quality_path = args.output / "calidad_por_columna.csv"
    metrics_path = args.output / "resumen_calidad.csv"
    chart_path = args.output / "panel_csv.png"
    column_report.to_csv(quality_path, index=False, encoding="utf-8-sig")
    metrics_report = pd.DataFrame({
        "indicador": [name for name, _ in metrics],
        "valor": pd.Series([value for _, value in metrics], dtype=object),
    })
    metrics_report.to_csv(
        metrics_path, index=False, encoding="utf-8-sig"
    )
    create_dashboard(
        df, missing, missing_cells, rows_with_missing, duplicate_ids,
        chronology_issues, csv_path.name, chart_path,
    )

    print(f"CSV analizado: {csv_path}")
    print(f"Registros: {len(df):,} | Columnas: {len(df.columns):,}")
    print(f"Celdas vacías: {missing_cells:,} | Filas con algún vacío: {rows_with_missing:,}")
    print(f"IDs repetidos: {duplicate_ids:,} | Fechas fuera de orden: {chronology_issues:,}")
    print(f"Panel: {chart_path.resolve()}")
    print(f"Calidad por columna: {quality_path.resolve()}")
    print(f"Resumen de calidad: {metrics_path.resolve()}")


if __name__ == "__main__":
    main()
