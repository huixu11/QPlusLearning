# Third-party provenance

Only the files needed to rebuild and check Lab 1 are vendored. The notebook installs the full pinned Kev checkout at runtime.

| Source | Pinned revision | Included files | License |
| --- | --- | --- | --- |
| [codaaiteam/jev-pacman](https://github.com/codaaiteam/jev-pacman/tree/8446fe74690cd61909bda91acfadbccb0f02b422) | `8446fe74690cd61909bda91acfadbccb0f02b422` | Original `index.html` and license | [MIT](jev-pacman/LICENSE), copyright codaaiteam |
| [jaredpalmer/kev](https://github.com/jaredpalmer/kev/tree/84847f0a883d900f7de5b7a57eaa341ca7f9a6b4) | `84847f0a883d900f7de5b7a57eaa341ca7f9a6b4` | Published `experiments/q35-08b.json` and license | [Apache-2.0](kev/LICENSE) |

The vendored files are unchanged. The derived [browser game](../games/pacman.html) changes the controllers and interface: Kev chooses Pac-Man's legal moves and ghosts follow deterministic rules. It preserves the community maze, renderer and full MIT notice. The notebook also bundles that notice with the community source.

Model weights and training suites are downloaded separately under their upstream terms. [upstream.json](../upstream.json) records their revisions. Comparison sources are linked in the slide speaker notes; their repositories and the supplied CLM paper are not redistributed here.
