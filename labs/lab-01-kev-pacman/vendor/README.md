# Third-party provenance

Only the files needed to rebuild and check Lab 1 are vendored. The notebook installs the full pinned Kev checkout at runtime.

| Source | Pinned revision | Included files | License |
| --- | --- | --- | --- |
| [masonicGIT/pacman](https://github.com/masonicGIT/pacman/tree/7407174c1d6a38be8cd230577489e39e0873145b) | `7407174c1d6a38be8cd230577489e39e0873145b` | Unchanged engine, source, assets and notices in [source.zip](arcade-pacman/source.zip); hashes in [source.json](arcade-pacman/source.json) | GPL-3.0; font notices included |
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev/tree/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4) | `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4` | Published `experiments/q35-4b-s23.json`, `experiments/night2-4b-du.json` (plus historical 0.8B recipes), `kev/train.py` and license | [Apache-2.0](kev/LICENSE) |

The vendored files are unchanged. The ZIP retains upstream LICENSE/COPYING, font notices, readable `src/` files and `build.sh`. QPlusLearning's GPL-3.0 game integration exposes state/player controls inside the native engine closure and adds a Colab shell. Classic mode keeps native drawing, font/audio, maze, four ghosts, timers, speeds, pellets, fruit, tunnels, collisions, lives and levels. Browser and CPU evaluation execute the same engine. The upstream author documents small differences from the arcade ROM.

Integration files: `games/arcade-engine.js`, `games/arcade-worker.js`, `games/arcade-browser.js`, `games/arcade-shell.html`. The browser embeds original font/audio assets for Colab output. Python helpers and Kev notices remain readable in the course repository.

Model weights and training suites are downloaded separately under their upstream terms. [upstream.json](../upstream.json) records their revisions. Comparison sources are linked in the slide speaker notes; their repositories and the supplied CLM paper are not redistributed here.
