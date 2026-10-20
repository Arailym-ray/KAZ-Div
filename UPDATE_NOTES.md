# KAZ-Div GitHub update package — 20 October 2026

Prepared files for the repository root:

- `README.md` — nine research components, accurate method-stage ordering (parsing-stage exact deduplication before filtering), task-specific recovery, Zenodo S1–S20, interpretation caveats.
- `CITATION.cff` — author order aligned to the revised manuscript: Arailym Tleubayeva, Zhansaya Makhambetova, Altynbek Seitenov, Pinar Sarisaray Boluk. The old version incorrectly listed Aigerim Mansurova. The existing software version 1.0.0 and original software release date are intentionally unchanged; the Zenodo DOI refers to the dataset/supplementary deposit, not automatically to the software release.
- `KAZ-Div_MANIFEST_latest.csv` — 297 records of the updated local `KAZ-Div_Supplementary_Materials_UPDATED_2026-10-20.zip`, relative to the `KAZ-Div_Supplementary_Materials/` directory, with size, SHA-256 and file provenance. All 297 hashes were verified against the updated local ZIP. Its internal `MANIFEST.csv` is intentionally excluded to avoid self-reference.

## How to apply

1. Open https://github.com/Arailym-ray/KAZ-Div on the `main` branch.
2. Replace these three files **at the repository root** with the contents in this package; keep their exact filenames.
3. Commit the changes, e.g. `docs: sync README, citation and manifest with October 2026 supplement`.
4. Verify links, `CITATION.cff` author consent/order, and the public Zenodo version's file checksums.

## Publication and provenance caveats

- The Zenodo update date (20 Oct 2026) and the revised Data S1–S4 / Tables S1–S20 scope are reported by the author. Hash validation was performed against the **local revised ZIP**, not a fresh download from Zenodo. The list is not a claim that every file has been independently verified on Zenodo.
- The new `CITATION.cff` is a **proposed replacement** until its author list is approved and committed to GitHub.
- This operation does **not** delete existing GitHub files and does not publish anything to GitHub. The previously attempted connector write was rejected with HTTP 403 (Resource not accessible by integration).
- If supplementary data change, regenerate the manifest instead of manually editing checksum values.
