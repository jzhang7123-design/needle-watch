# Needle photo watcher: setup guide (about 10 minutes, free)

**What it does:** every ~10 minutes a free GitHub server checks the site's public photo list once. For each new photo it downloads the small thumbnail, records the time it first saw it and its brightness, and saves that to `data/photos.csv` in your repo. Your laptop can be off. It never logs in, submits guesses, or buys anything.

**Only use it if the site owners said it's OK.** It will not run until you confirm that (step 3).

## Setup
1. Create a free account at github.com, then a new **private** repository (e.g. `needle-watch`).
2. Upload everything in this zip to the repo, keeping the folders (`watcher/`, `.github/workflows/watch.yml`). On the repo page: Add file > Upload files. Make sure the hidden `.github` folder is included (drag the whole folder).
3. In the repo go to **Settings > Secrets and variables > Actions**:
   - *Optional:* tab *Secrets* > New repository secret: name `CONTACT_EMAIL`, value your email (added to the request's User-Agent so the site can reach you). Skip it if the site said it isn't needed.
   - tab *Variables* > New repository variable: name `PERMISSION_CONFIRMED`, value `yes`. Only set this if the site really agreed.
4. In **Settings > Actions > General**, set Workflow permissions to *Read and write*.
5. Open the **Actions** tab, choose *needle-watch*, click *Run workflow*. Check the log. The first run records the existing photos as "backfill" (numbers only, no times), which is expected.
6. Done. It now runs by itself. Later, download `data/photos.csv` and upload it to the Colab notebook (Option C).

## Things to know (honest limits)
- **Times are "first seen", not release or capture times.** They lag by the polling interval plus GitHub's scheduling delay (often 5 to 15 minutes, sometimes more). In the notebook use `MAX_DELAY_MIN = 30` or more.
- **Photos released before the first run have no time.** They go to `data/backfill.csv`. Keep typing their release times by hand if you want them in the analysis. The watcher's own data needs a few days to build up.
- **After an outage** (or a burst of many photos at once) new photos are marked backfill instead of live, so unreliable times never get into the analysis.
- **GitHub can skip scheduled runs** under load, and turns off scheduled workflows in a repo with no activity for 60 days. The watcher's own commits count as activity, so it normally keeps itself alive. If it stops, re-enable it in the Actions tab.
- **If the site changes its photo list format,** the run fails with exit code 4 and saves `data/raw_images_sample.txt`. Send me that file and I'll adapt the parser. I could not see the real format from here, so the first real run is the real test.
- **If the site says no (HTTP 401/403),** the watcher stops immediately. Don't work around it. Contact the site.
- Change the schedule in `.github/workflows/watch.yml` (`*/10` means every 10 minutes). Don't go below 5, and don't poll faster than the site's owners agreed to.

## Run it by hand
```
CONTACT_EMAIL=you@example.com PERMISSION_CONFIRMED=yes python watcher/watch.py
```
Needs Python 3.10+ with `pillow numpy pandas`.
