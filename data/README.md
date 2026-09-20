
# Data preparation

SIGNAL uses preprocessed user–item interactions, item-aligned modality features, and an item co-occurrence graph.

## Datasets in the paper

Statistics below are from Table 3 of the SIGNAL paper.

| Dataset | Users | Items | Interactions | Item features |
| :--- | ---: | ---: | ---: | :--- |
| Baby | 19,445 | 7,050 | 160,792 | Visual: 4096; textual: 384 |
| Sports | 35,598 | 18,357 | 296,337 | Visual: 4096; textual: 384 |
| Clothing | 39,387 | 23,033 | 278,677 | Visual: 4096; textual: 384 |
| TikTok | 9,308 | 4,710 | 68,722 | Visual: 128; textual: 768; acoustic: 128 |

The paper uses an 80% / 10% / 10% train / validation / test split and 5-core filtering for the Amazon datasets.

## Existing MMRec resources

The inherited MMRec data documentation provides a [Baby / Sports / Elec Google Drive folder](https://drive.google.com/drive/folders/13cBy1EA_saTUuXxVllKgtfci2A09jyaG?usp=sharing). Elec is not one of the four datasets reported in the SIGNAL paper.

The original [MicroLens resource](https://drive.google.com/drive/folders/14UyTAh_YyDV8vzXteBJiy9jv8TBDK43w?usp=drive_link) is a separate dataset; it is not the paper's TikTok dataset.

## Expected layout

For Baby, the supplied dataset configuration expects:

```text
data/
└── baby/
    ├── baby.inter
    ├── image_feat.npy
    ├── text_feat.npy
    └── <item_graph_dict_file>.npy
```

`<item_graph_dict_file>.npy` denotes the graph filename specified by the `item_graph_dict_file` configuration key; it is not a literal filename to create.

The tab-separated interaction file uses `userID`, `itemID`, and `x_label`. The loader interprets `x_label` values `0`, `1`, and `2` as train, validation, and test. User and item IDs must be zero-based consecutive integers, and feature row indices must match item IDs. See [the Baby configuration](../src/configs/dataset/baby.yaml) and [dataset loader](../src/utils/dataset.py).

SIGNAL loads the item graph with `np.load(..., allow_pickle=True).item()`. Its `get_session_adj()` method expects an item-ID-keyed dictionary with neighbor IDs at entry `[0]` and corresponding weights at entry `[1]`. The current method builds the normalized adjacency from the neighbor indices; its collected weights are not passed to normalization.

## Co-occurrence graph

The paper constructs this graph from **training interactions only**, with pruning and top-k selection described in Section 3.2.1. Validation and test interactions must be excluded.

The inherited `dualgnn-gen-u-u-matrix.py` creates a **user–user** graph for DualGNN. It does not supply SIGNAL's required **item–item** co-occurrence graph.

## Preprocessing

The [inherited preprocessing tools](../preprocessing/README.md) cover Amazon interaction conversion, splitting, feature reindexing, and feature encoding.

See [the experiment settings](../docs/REPRODUCIBILITY.md) for the paper’s hyperparameters and configuration conventions.
