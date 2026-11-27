# Kairos pretraining datasets (mirror)

Parquet copies of the Hugging Face datasets, re-sharded below GitHub's 100 MB file limit:
- `keep-it-simple/` from `ffurfaro/keep-it-simple`
- `keep-it-simple-multimodal/` from `ffurfaro/keep-it-simple-multimodal`

Text dataset: all rows. Check the licences of the upstream sources
(Flickr8k, AudioCaps, Molmo2-VideoCapQA, nuScenes-mini, MotionSense, PixelBytes-OptimalControl) before reusing.

Load with `datasets.load_dataset('<folder>', split='train')` or `python scripts/fetch_datasets.py --from-github <user>/<repo>`.
