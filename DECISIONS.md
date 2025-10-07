# Decisions and Notes

## Purpose
Build a Streamlit app to upload jewelry images, embed using DINOv2 ViT-B/14 (768-d) on GPU if available, store vectors in Qdrant, and perform similarity search with cosine distance.

## Key Choices
- Model: `dinov2_vitb14(pretrained=True)` with `.eval()` on `cuda` if available, else CPU.
- Embedding size: 768. All vectors L2-normalized prior to upsert/search to align with cosine distance.
- Qdrant: Connect via URL (default `http://localhost:6333`), collection `jewelry_images`, `VectorParams(size=768, distance=cosine)`.
- Storage: Save all uploads in local `images/` for display consistency and recoverability.
- Duplicate handling: Check by `filename` existence in Qdrant; also compute SHA1 for metadata traceability.
- Item type inference: Heuristic from filename keywords (`ring`, `necklace`, `bracelet`, `earring`), fallback to `unknown`.
- UI: Two tabs — Upload and Search. Sidebar for settings (Qdrant URL, collection, Top-K). Progress bars and spinners for long ops.

## Error Handling
- Qdrant connection failures surfaced early with `st.error` and disable actions accordingly.
- Duplicate uploads raise `FileExistsError` and report via status area.
- Missing local image files during display show a warning while keeping results.

## Preprocessing
- Convert to RGB, resize so shortest side is 518, center-crop 518×518, normalize using ImageNet mean/std, feed through DINOv2, take global embedding. Returned as `float32` numpy, unit length.

## Running
- Install dependencies on a GPU-enabled host.
- Start: `streamlit run app.py`.
- Qdrant must be running locally at the configured URL.

## Future Improvements
- Persist id as SHA1 of file bytes for stronger duplicate semantics.
- Add upload of labels/types explicitly from user input.
- Batch upsert for higher throughput.
- Preview and remove items from the collection.
