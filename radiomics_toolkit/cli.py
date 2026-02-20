"""Click-based CLI for the radiomics toolkit.

Usage:
    radiomics extract --image <path> [--mask <path>] --output <file> [options]
"""

from __future__ import annotations

import sys

import click

from .extractor import RadiomicsExtractor, _ALL_CLASSES
from .io.writers import write_results


@click.group()
def cli() -> None:
    """Radiomics Toolkit — quantitative imaging feature extraction."""


@cli.command("extract")
@click.option(
    "--image", "-i", required=True,
    type=click.Path(exists=True),
    help="Path to image file or DICOM directory.",
)
@click.option(
    "--mask", "-m", default=None,
    type=click.Path(exists=True),
    help="Path to binary mask file (optional; defaults to whole image).",
)
@click.option(
    "--output", "-o", required=True,
    type=click.Path(),
    help="Output file path (.csv or .json).",
)
@click.option(
    "--bins", default=64, show_default=True,
    type=int,
    help="Number of grey-level bins for discretization.",
)
@click.option(
    "--features", default=",".join(_ALL_CLASSES), show_default=True,
    help="Comma-separated list of feature classes to compute.",
)
@click.option(
    "--normalize", is_flag=True, default=False,
    help="Normalize image intensities to [0, 1] before extraction.",
)
def extract_cmd(
    image: str,
    mask: str | None,
    output: str,
    bins: int,
    features: str,
    normalize: bool,
) -> None:
    """Extract radiomics features from IMAGE and write to OUTPUT."""
    feature_classes = [f.strip() for f in features.split(",") if f.strip()]

    click.echo(f"Loading image: {image}")
    if mask:
        click.echo(f"Loading mask:  {mask}")

    try:
        extractor = RadiomicsExtractor(
            bin_count=bins,
            feature_classes=feature_classes,
            normalize=normalize,
        )
        results = extractor.extract_from_file(image, mask_path=mask)
    except Exception as exc:
        click.echo(f"Error during extraction: {exc}", err=True)
        sys.exit(1)

    click.echo(f"Extracted {len(results)} features.")

    try:
        write_results(results, output)
        click.echo(f"Results written to: {output}")
    except Exception as exc:
        click.echo(f"Error writing output: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
