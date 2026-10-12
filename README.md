# Fashion Retrieval

Image-based fashion search. Give it a clothing photo and it returns the most visually similar items from a product catalog, using CLIP image embeddings and cosine-similarity ranking. The catalog is a scraped Uniqlo US product set (5,094 images across 917 products), with the larger Fashion200k dataset (201,624 images) staged in a Google Cloud Storage bucket for scale-up.

The retrieval core is small on purpose: embed every catalog image once, cache the vectors, and rank by dot product against the query embedding.

## Status

| Piece | State |
| --- | --- |
| Uniqlo catalog data (scraped, cleaned, filtered for loadable images) | Done |
| Image embeddings with OpenCLIP ViT-B/32, cached to disk | Done |
| Image-to-image retrieval and result visualization | Done |
| Image pipeline into a GCS bucket (Uniqlo and Fashion200k) | Done |
| Text-query retrieval | Planned |
| Fashion-specific model comparison (Marqo-FashionCLIP vs. CLIP) | Planned |
| Quantitative evaluation (precision@k, recall@k) | Planned |
| Embeddings and retrieval over Fashion200k | Planned |

Only the "Done" rows are in this repo today. Retrieval quality has been checked by eye on sample queries, not yet measured.

## How it works

```
 Uniqlo product pages          Marqo/fashion200k (Hugging Face)
        |                                   |
   scraped CSV                              |
        v                                   v
  clean + explode to one row          parallel upload
  per image, drop bad URLs            (200-image batches)
        |                                   |
        +-----------> GCS bucket <----------+
                          |
        OpenCLIP ViT-B/32 image encoder  (512-d, L2-normalized)
                          |
                 cached feature matrix  (features-cache/uniqlo_vit32.pkl)
                          |
   query image --> encode --> cosine similarity --> top-k neighbors
```

1. **Catalog cleaning** (`data-analysis/uniqlo_dataset.ipynb`). The raw scrape has one row per product (917 rows) with up to 11 image URL columns. These are exploded into one row per image (5,329 rows), each image URL is test-loaded, and a boolean `is_good` flag is stored. 5,094 images load.
2. **Storage** (`store_in_gcp.ipynb`, `data-analysis/fashion200k_dataset.ipynb`). Uniqlo images are saved to a GCS bucket as `uniqlo_data/{product_id}_{local_id}.png` (5,096 of 5,329 uploaded; 233 failed). Fashion200k is pulled from Hugging Face and uploaded in parallel with a thread pool, about 201.5k objects under `fashion200k/`. The cached feature file is also stored in the bucket.
3. **Embedding** (`rice.py`). The `RICES` class loads OpenCLIP `ViT-B-32` (OpenAI weights), encodes every image in batches, L2-normalizes the vectors, and returns a `[N, 512]` tensor. The full Uniqlo run took about 13 minutes in the notebook and is cached in `features-cache/uniqlo_vit32.pkl`.
4. **Retrieval** (`RICES.find`). The query image is encoded the same way, scored against all cached vectors with one matrix multiply (equivalent to cosine similarity because everything is normalized), and the top-k indices are returned.

`RICES` stands for Retrieval-based In-Context Example Selection, the name this retrieval pattern goes by in the vision-language literature. Here it is used directly as a similarity-search index.

## Data

| Dataset | Size | Where | Used for |
| --- | --- | --- | --- |
| Uniqlo US catalog | 917 products, 5,329 images (5,094 loadable) | `data/uniqlo_data.csv`, `data/uniqlo_processed.csv` | Embedding and retrieval |
| [Marqo/fashion200k](https://huggingface.co/datasets/Marqo/fashion200k) | 201,624 images with `category1/2/3`, text description, `item_ID` | GCS bucket, `fashion200k/` | Staged for scale-up |

Uniqlo image rows by department: Women 1,922, Men 1,552, Unisex 1,106, Kids 749. Each row carries `id`, `price`, `item_url`, `Item_name`, `Item_type` (department/category/subcategory), `materials`, and `img_url`.

## Repository layout

```
rice.py                       OpenCLIP encoder + feature cache + top-k search (RICES)
dataloader/dataloader.py      UniqloDataset: PyTorch Dataset over the cleaned CSV
feature_computer.ipynb        Compute/load features, run queries, plot results
data-analysis/                Catalog cleaning and Fashion200k exploration
store_in_gcp.ipynb            Upload Uniqlo images and features to GCS
experimental/                 Scratch notebooks (bucket I/O)
get_first_n_images.ipynb      Save the first N images locally with metadata
visualize_multiple_images.ipynb  Grid viewer for saved images
data/                         Raw and processed Uniqlo CSVs
features-cache/               Cached ViT-B/32 embeddings (5,094 x 512)
```

## Quickstart

```bash
git clone https://github.com/AbhishekSureddy/fashion-retrieval
cd fashion-retrieval
pip install torch torchvision open_clip_torch pandas pillow requests tqdm matplotlib
# only needed for the GCS and Fashion200k notebooks:
pip install google-cloud-storage datasets
```

`requirements.txt` is a full export of the original Jupyter/conda environment and will not install cleanly with pip. The line above is the minimal set the retrieval code needs.

Search with the cached embeddings (no GPU needed):

```python
import pickle
from rice import RICES
from dataloader.dataloader import UniqloDataset

dataset = UniqloDataset("./data/uniqlo_processed.csv")   # downloads each image on access
with open("features-cache/uniqlo_vit32.pkl", "rb") as f:
    features = pickle.load(f)

retriever = RICES(dataset, device="cpu", batch_size=128, cached_features=features)

query = dataset[0]["image"]
neighbors = retriever.find([query], num_examples=10)[0]
```

`find` returns neighbors with the **most similar last**. Because the query comes from the catalog, it is its own top hit; `feature_computer.ipynb` reverses the list and drops that first result before plotting. To re-embed from scratch, omit `cached_features` and pass `device="cuda"` if you have a GPU.

## Known limitations

- **Images are fetched live.** `UniqloDataset.__getitem__` downloads from the Uniqlo CDN on every access, so it is slow and breaks if image URLs change. Reading from the GCS copy would fix both.
- **Several variants of the same product.** Each product has multiple images (colors and angles), so a query's nearest neighbors are often other photos of the same item. That is expected, but it matters for how relevance should be defined when evaluating.
- **No metrics yet.** Results are judged qualitatively in `feature_computer.ipynb`.
- **Notebook-centric.** The workflow lives in notebooks with some hard-coded paths and a GCP credentials path; there is no CLI or package yet.

## Roadmap

1. **Text queries.** Encode a text query with the same CLIP model and rank against the image embeddings (CLIP's shared image-text space makes this a small change to `RICES.find`).
2. **Model comparison.** Swap in Marqo-FashionCLIP and compare it with the general-purpose ViT-B/32 on the same queries.
3. **Evaluation.** Define relevance (for Fashion200k, shared `category2`/`category3` labels or the text description; for Uniqlo, same product or same `Item_type`) and report precision@k and recall@k for each model.
4. **Scale-up.** Embed Fashion200k from the bucket and move to an approximate-nearest-neighbor index once the catalog is too large for a single matrix multiply.
5. **Serving.** Wrap retrieval in a small API and add a minimal search UI.

## Acknowledgements

Embeddings use [OpenCLIP](https://github.com/mlfoundations/open_clip). Fashion200k is distributed by [Marqo](https://huggingface.co/datasets/Marqo/fashion200k) on Hugging Face. Product images and metadata are from Uniqlo's public website and are used here for research and demonstration only.
