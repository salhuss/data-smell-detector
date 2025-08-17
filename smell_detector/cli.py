# smell_detector/cli.py
import typer
from pathlib import Path
import pandas as pd

from .checks import run_checks
from .export import write_json, render_html

app = typer.Typer(help="Data Smell Detector CLI")

@app.command()
def main(
    data: str = typer.Argument(..., help="Path to CSV or Parquet (current dataset)"),
    target: str = typer.Option(None, "--target", help="Name of target/label column (optional)"),
    out: str = typer.Option("out", help="Output directory"),
    rows: int = typer.Option(100000, help="Max rows to read from each dataset for speed"),
    ref: str = typer.Option(None, "--ref", help="Optional reference CSV/Parquet for drift & schema diff"),
    save_plots: bool = typer.Option(True, "--save-plots/--no-save-plots", help="Save correlation heatmap PNG"),
):
    """
    Scan a dataset for common 'data smells' and (optionally) compare to a reference dataset
    for drift and schema differences. Exports HTML+JSON.
    """
    out_p = Path(out); out_p.mkdir(parents=True, exist_ok=True)

    def _load(path: str) -> pd.DataFrame:
        p = Path(path)
        if p.suffix.lower() in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        return df

    # Load current
    df = _load(data)
    if rows and len(df) > rows:
        df = df.sample(rows, random_state=42).reset_index(drop=True)

    # Load reference if provided
    ref_df = None
    if ref:
        ref_df = _load(ref)
        if rows and len(ref_df) > rows:
            ref_df = ref_df.sample(rows, random_state=42).reset_index(drop=True)

    # Run checks (+ drift/schema if ref provided)
    report = run_checks(df, target=target, ref_df=ref_df, out_dir=str(out_p), save_plots=save_plots)

    # Export
    write_json(out_p / "report.json", report)
    render_html(out_p / "report.html", report)

    typer.echo(f"✅ Report written → {out_p.resolve()}/report.html")
    typer.echo("   JSON          → report.json")
    if save_plots and report.get("plots", {}).get("corr_heatmap_png"):
        typer.echo(f"   Plot          → {report['plots']['corr_heatmap_png']}")

if __name__ == "__main__":
    app()
