# Acknowledgements

SIGNAL is based on [MMRec](https://github.com/enoche/MMRec), developed by Xin Zhou and contributors. The shared training loop, evaluation, data handling, configuration system, preprocessing tools, and baseline models originate from that framework. Original source headers and the GPL v3 license are retained.

This repository is organized around the SIGNAL paper and its implementation in [src/models/signal.py](src/models/signal.py). The upstream model catalogue is archived in [docs/upstream/MMRec-README.md](docs/upstream/MMRec-README.md).

If you use the MMRec infrastructure in your research, please also acknowledge the framework:

```bibtex
@inproceedings{zhou2023mmrec,
  title     = {Mmrec: Simplifying multimodal recommendation},
  author    = {Zhou, Xin},
  booktitle = {Proceedings of the 5th ACM International Conference on Multimedia in Asia Workshops},
  pages     = {1--2},
  year      = {2023}
}
```

## Source provenance

The starting point for this repository reorganization was `Sail-high/SIGNAL`, commit `83a52a4` ("Rename signal.py to MMRec-master/src/models/signal.py"). The nested `MMRec-master/` contents were moved to the repository root; its original README, images, and report were preserved under `docs/upstream/`.

The SIGNAL model's two shared-module imports were updated to match the existing `common` package. The entry point now defaults to SIGNAL. The model's mathematical operations were not changed during this documentation and repository update.

## Paper and figure

The paper metadata, benchmark tables, and framework figure come from the supplied published article, DOI [10.1016/j.eswa.2026.134101](https://doi.org/10.1016/j.eswa.2026.134101). `docs/assets/framework.png` reproduces Figure 2. Publication notices in the supplied PDF are retained; the code license does not replace those notices.
