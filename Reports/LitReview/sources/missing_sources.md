# Missing / Non-Public Sources

The literature review currently cites **39** sources (see `all_sources.csv` for the full inventory). Some are books, some are paywalled journal/conference papers, and some are public-domain PDFs.

I only download sources from official/public locations. I do not pull books or paywalled PDFs from unofficial mirrors.

## Inventory of what's tracked
- Full inventory (all 39 bibitems): `all_sources.csv`
- Public download list: `manifest.csv`
- Downloaded PDFs: `downloaded/`

## How to complete the set
1. For paywalled items, download using university library access (IEEE, AIAA, publisher portals) and place PDFs in `downloaded/`.
2. For books, use library e-books or scanned excerpts you are allowed to use, and place them in `downloaded/` (or keep as external references if you do not want copies in-repo).
3. Re-run `compile.ps1` after adding PDFs if you want to keep the local archive in sync.

If you want, I can add a second manifest (`manifest_paywalled.csv`) that records DOI/publisher landing pages for the paywalled items so nothing is untracked.
