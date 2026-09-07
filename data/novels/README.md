# Source novels

Download a Project Gutenberg edition as `<NOVEL_NAME>.txt`, then split it into
chapters. Run these commands from the release root:

~~~bash
python -m arc_construction.download_novels --novel Anna_Kareina

NOVEL_NAME=Anna_Kareina uv run python -m arc_construction.phase0_preprocess
~~~

Use `--all` for all eighteen public-domain titles, `--list` for the source
catalog, or repeat `--novel` to select several titles. Quote slugs containing
spaces, for example `--novel "Great Expectations"`.

The downloader uses [Project Gutenberg's own mirror](https://gutenberg.pglaf.org/),
listed in its [mirror directory](https://www.gutenberg.org/MIRRORS.ALL), following
the [automated-download guidance](https://www.gutenberg.org/policy/robot_access.html).
It checks the ebook ID, UTF-8 encoding, and Gutenberg boundary markers before
saving, prints each file's SHA-256 hash, and never replaces an existing file.
Gutenberg editions can be updated; retain the downloaded files and their hashes
when repeating a run. The download command requires no API key or additional
Python packages.

Downloaded source files and generated chapters remain local and are excluded
from version control and release packages.

Phase 0 needs no API key and writes `results/arc_extraction/<NOVEL_NAME>/chapters/`,
which is not shipped and which later stages read. `NOVEL_NAME` must match the
directory name under `results/arc_extraction/` exactly — the shipped arcs and
probes are keyed by that slug.

## The 19 novels of the paper

Eighteen are Project Gutenberg editions in the U.S. public domain and the
identifier is the ebook number. Harry Potter is the one title still under
copyright; its source text and benchmark data are not distributed.

### Validated evaluation slice (paper Table 2)

| Novel | `NOVEL_NAME` | Source |
|---|---|---|
| Anna Karenina | `Anna_Kareina` | Project Gutenberg [#1399](https://www.gutenberg.org/ebooks/1399) |
| The Count of Monte Cristo | `Monte_Cristo` | Project Gutenberg [#1184](https://www.gutenberg.org/ebooks/1184) |
| Don Quixote | `don_quixote` | Project Gutenberg [#996](https://www.gutenberg.org/ebooks/996) |
| The Autobiography of Benjamin Franklin | `Benjamin_Franklin` | Project Gutenberg [#20203](https://www.gutenberg.org/ebooks/20203) |
| Harry Potter (Books 1–7) | `Harry_Potter` | **not Gutenberg, not redistributed** — rebuild from a legally obtained copy |

### Low-popularity evaluation slice (held out, unvalidated)

| Novel | `NOVEL_NAME` | Source |
|---|---|---|
| He Knew He Was Right | `He Knew He Was Right` | Project Gutenberg [#5140](https://www.gutenberg.org/ebooks/5140) |
| The Odd Women | `The Odd Women` | Project Gutenberg [#4313](https://www.gutenberg.org/ebooks/4313) |

### Training slice (SFT / DPO / RLVR)

| Novel | `NOVEL_NAME` | Source |
|---|---|---|
| Anne of Green Gables | `Anne of Green Gables` | Project Gutenberg [#45](https://www.gutenberg.org/ebooks/45) |
| The Picture of Dorian Gray | `Dorian Gray` | Project Gutenberg [#174](https://www.gutenberg.org/ebooks/174) |
| East Lynne | `East Lynne` | Project Gutenberg [#3322](https://www.gutenberg.org/ebooks/3322) |
| Great Expectations | `Great Expectations` | Project Gutenberg [#1400](https://www.gutenberg.org/ebooks/1400) |
| Jane Eyre | `Jane_Eyre` | Project Gutenberg [#1260](https://www.gutenberg.org/ebooks/1260) |
| Lady Audley's Secret | `Lady Audley's Secret` | Project Gutenberg [#8954](https://www.gutenberg.org/ebooks/8954) |
| Little Women | `Little Women` | Project Gutenberg [#37106](https://www.gutenberg.org/ebooks/37106) |
| The Interesting Narrative of the Life of Olaudah Equiano | `Olaudah Equiano` | Project Gutenberg [#15399](https://www.gutenberg.org/ebooks/15399) |
| Persuasion | `Persuasion` | Project Gutenberg [#105](https://www.gutenberg.org/ebooks/105) |
| Pride and Prejudice | `pride_and_prejudice` | Project Gutenberg [#1342](https://www.gutenberg.org/ebooks/1342) |
| The Underdogs | `The Underdogs` | Project Gutenberg [#549](https://www.gutenberg.org/ebooks/549) |
| Treasure Island | `Treasure Island` | Project Gutenberg [#120](https://www.gutenberg.org/ebooks/120) |

`East Lynne` and `The Underdogs` are in the training pool, so ArcANE results on
them are in-distribution. The paper reports them in a separate reference table
(`--novels extra3`); they are **not** the low-popularity control.

## Chapter headings

Phase 0 strips the Project Gutenberg header and footer, then keeps whichever of
six heading patterns matches most often: `Chapter 1`, `CHAPTER I.`, a roman
numeral alone on a line (`IV`), `CHAPTER ONE`, `Chapter Twenty-Three`, and an
indented roman numeral (`    XI.`). Add a pattern to `_CHAPTER_PATTERNS` in
[`arc_construction/phase0_preprocess.py`](../../arc_construction/phase0_preprocess.py) for a text these do not cover; chapters
over 200k characters are sub-chunked and fragments under 300 characters are
dropped as table-of-contents artifacts.
