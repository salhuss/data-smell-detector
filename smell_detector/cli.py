import typer
from pathlib import Path
import pandas as pd

from .checks import run_checks
from .export import write_json, render_html

app = typer.Typer(help="Data Smell Detector CLI")

@app.command()
def main(
    data: str = typer.Argument(..., help="Path to CSV or Parquet"),
    target: str = typer.Option(None, "--target", help="Name of target/label column (optional)"),
    out: str = typer.Option("out", help="Output directory"),
    rows: int = typer.Option(100000, help="Max rows to read for speed"),
):
    """
    Scan a dataset for common 'data smells' (missingness, skew, imbalance, duplicates,
    potential leakage, high cardinality, constant columns, outliers) and export HTML+JSON.
    """
    out_p = Path(out); out_p.mkdir(parents=True, exist_ok=True)

    # Load
    p = Path(data)
    if p.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)
    if rows and len(df) > rows:
        df = df.sample(rows, random_state=42).reset_index(drop=True)

    # Run checks
    report = run_checks(df, target=target)

    # Export
    write_json(out_p / "report.json", report)
    render_html(out_p / "report.html", report)

    typer.echo(f"✅ Report written → {out_p.resolve()}/report.html")
    typer.echo( "   JSON          → report.json")

if __name__ == "__main__":
    app()
