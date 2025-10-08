"""
Dependencies (install on the target machine with GPU):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install dinov2 qdrant-client Pillow numpy streamlit

Run the app:
    streamlit run app.py
"""

import io
import os
import time
import hashlib
import warnings
from datetime import datetime
from typing import List, Tuple

import numpy as np
from PIL import Image

import streamlit as st

import torch

# Suppress xFormers warnings from DINOv2
warnings.filterwarnings("ignore", message="xFormers is not available")

# Try to import DINOv2 from the package; if unavailable, fall back to torch.hub
try:
    from dinov2.models import dinov2_vitb14 as _dinov2_ctor  # type: ignore

    def _create_dino_model():
        return _dinov2_ctor(pretrained=True)
except Exception:
    def _create_dino_model():
        # Loads model weights from the official repo via torch.hub
        # Requires internet access on first run; cached afterward
        return torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)


# -----------------------------
# Configuration & constants
# -----------------------------
DEFAULT_COLLECTION = "jewelry_images"
DEFAULT_TOP_K = 5
DEFAULT_QDRANT_HOST = "http://localhost:6333"
IMAGES_DIR = "images"


# Ensure images directory exists
os.makedirs(IMAGES_DIR, exist_ok=True)


# -----------------------------
# Sidebar settings
# -----------------------------
st.set_page_config(page_title="Jewelry Image Similarity Search (DINOv2 + Qdrant)", layout="wide")

with st.sidebar:
    st.header("Settings")
    qdrant_url = st.text_input("Qdrant URL", value=DEFAULT_QDRANT_HOST, help="e.g., http://localhost:6333")
    collection_name = st.text_input("Collection Name", value=DEFAULT_COLLECTION)
    top_k_setting = st.number_input("Top K Results", min_value=1, max_value=50, value=DEFAULT_TOP_K, step=1)
    st.markdown("---")
    st.caption("Embeddings generated with DINOv2 ViT-B/14 (768-d) on GPU if available.")


# -----------------------------
# Model loading (GPU if available)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _create_dino_model().to(device)
    model.eval()
    return model, device


model, device = load_model_and_device()


# -----------------------------
# Qdrant client and collection setup
# -----------------------------
@st.cache_resource(show_spinner=True)
def get_qdrant_client(url: str) -> QdrantClient:
    # qdrant-client supports passing URL directly
    return QdrantClient(url=url)


def ensure_collection_exists(client: QdrantClient, name: str, vector_size: int = 768):
    try:
        # Prefer collection_exists if available, else fallback to get_collection
        exists = False
        try:
            exists = client.collection_exists(name)
        except AttributeError:
            # Older clients may not have collection_exists
            try:
                client.get_collection(name)
                exists = True
            except Exception:
                exists = False

        if not exists:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
    except Exception as e:
        raise RuntimeError(f"Failed to ensure collection '{name}' exists: {e}")


@st.cache_resource(show_spinner=True)
def init_qdrant(url: str, name: str) -> QdrantClient:
    client = get_qdrant_client(url)
    ensure_collection_exists(client, name, vector_size=768)
    return client


# -----------------------------
# Image preprocessing and embedding
# -----------------------------
def _preprocess_image_for_dino(image: Image.Image) -> torch.Tensor:
    # Convert to RGB and resize proportionally with side >= 518 (DINOv2 default crops ~518)
    # Then center-crop to 518 and normalize with ImageNet mean/std
    image = image.convert("RGB")

    # Keep aspect, make shortest side 518
    min_side = 518
    w, h = image.size
    scale = min_side / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop 518x518
    left = (image.width - min_side) // 2
    top = (image.height - min_side) // 2
    image = image.crop((left, top, left + min_side, top + min_side))

    img = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    # HWC -> CHW
    img = np.transpose(img, (0, 1, 2))  # keep HWC first for clarity
    img = np.transpose(img, (2, 0, 1))
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    return tensor


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def get_embedding(image: Image.Image) -> np.ndarray:
    """Generate a 768-dim embedding from a PIL.Image using DINOv2 ViT-B/14.

    Returns a numpy array normalized to unit length.
    """
    with torch.no_grad():
        tensor = _preprocess_image_for_dino(image)
        # DINOv2 returns features via model(x). We use the global [CLS] representation.
        feats = model(tensor)
        # Convert to numpy
        emb = feats.detach().cpu().numpy().reshape(-1)
        emb = l2_normalize(emb)
        return emb.astype(np.float32)


# -----------------------------
# Qdrant upsert & search helpers
# -----------------------------
def _file_sha1(filepath: str) -> str:
    h = hashlib.sha1()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _already_indexed(client: QdrantClient, collection: str, filename: str) -> bool:
    try:
        flt = Filter(must=[FieldCondition(key="filename", match=MatchValue(value=filename))])
        points, _next = client.scroll(collection_name=collection, scroll_filter=flt, limit=1)
        return len(points) > 0
    except Exception:
        return False


def upsert_image(filepath: str):
    """Upsert a single image file into Qdrant with embedding and metadata.

    Metadata: filename, timestamp, type (inferred by folder or filename heuristic), sha1
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")

    filename = os.path.basename(filepath)

    # Duplicate check by filename in Qdrant
    if _already_indexed(qdrant_client, collection_name, filename):
        raise FileExistsError(f"Duplicate image upload detected for filename: {filename}")

    # Load image
    with Image.open(filepath) as img:
        embedding = get_embedding(img)

    # Simple heuristic for type from filename
    lower = filename.lower()
    if any(k in lower for k in ["ring", "band"]):
        item_type = "ring"
    elif any(k in lower for k in ["necklace", "chain", "pendant"]):
        item_type = "necklace"
    elif any(k in lower for k in ["bracelet", "bangle"]):
        item_type = "bracelet"
    elif any(k in lower for k in ["earring", "stud", "hoop"]):
        item_type = "earring"
    else:
        item_type = "unknown"

    # Build payload
    payload = {
        "filename": filename,
        "filepath": filepath,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": item_type,
        "sha1": _file_sha1(filepath),
    }

    # Upsert
    point = PointStruct(
        id=hashlib.md5(filename.encode("utf-8")).hexdigest(),
        vector=embedding.tolist(),
        payload=payload,
    )
    qdrant_client.upsert(collection_name=collection_name, points=[point])


def search_similar(image: Image.Image, top_k: int) -> List[Tuple[str, float, dict]]:
    """Search similar images and return list of (filepath, score, payload)."""
    query_vec = get_embedding(image)
    res = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_vec.tolist(),
        limit=top_k,
        with_payload=True,
    )
    results: List[Tuple[str, float, dict]] = []
    for point in res:
        payload = point.payload or {}
        path = payload.get("filepath") or os.path.join(IMAGES_DIR, payload.get("filename", ""))
        score = float(point.score)
        results.append((path, score, payload))
    return results


# -----------------------------
# Initialize Qdrant once settings are available
# -----------------------------
try:
    qdrant_client = init_qdrant(qdrant_url, collection_name)
    qdrant_ok = True
except Exception as e:
    qdrant_ok = False
    st.error(f"Failed to connect to Qdrant at {qdrant_url}: {e}")


# -----------------------------
# UI: Main sections
# -----------------------------
st.title("Jewelry Image Similarity Search")
st.caption("Upload jewelry images to build a database, then search by image using DINOv2 ViT-B/14 embeddings stored in Qdrant.")

tab_db, tab_search, tab_gallery = st.tabs(["Upload to Database", "Search by Image", "Gallery & Manage"])


# -----------------------------
# Upload to Database
# -----------------------------
with tab_db:
    st.subheader("Upload and Index Images")
    uploaded_files = st.file_uploader(
        "Upload jewelry images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )

    if st.button("Process & Upsert", disabled=not uploaded_files or not qdrant_ok):
        if not qdrant_ok:
            st.error("Qdrant connection is not available. Please check settings.")
        else:
            successes = 0
            duplicates = 0
            errors = 0
            progress = st.progress(0)
            status = st.empty()
            for idx, uf in enumerate(uploaded_files):
                try:
                    # Save file to images/ folder
                    filename = os.path.basename(uf.name)
                    save_path = os.path.join(IMAGES_DIR, filename)

                    if os.path.exists(save_path):
                        # If file exists locally, still attempt duplicate check in Qdrant
                        if _already_indexed(qdrant_client, collection_name, filename):
                            duplicates += 1
                            status.warning(f"Duplicate skipped: {filename}")
                            progress.progress(int(((idx + 1) / len(uploaded_files)) * 100))
                            continue

                    with open(save_path, "wb") as out:
                        out.write(uf.read())

                    # Upsert into Qdrant
                    with st.spinner(f"Embedding and upserting {filename}..."):
                        upsert_image(save_path)
                        successes += 1
                        status.info(f"Indexed: {filename}")
                except FileExistsError as de:  # duplicate upload
                    duplicates += 1
                    status.warning(str(de))
                except Exception as ex:
                    errors += 1
                    status.error(f"Error for {uf.name}: {ex}")
                finally:
                    progress.progress(int(((idx + 1) / len(uploaded_files)) * 100))

            st.success(f"Done. Indexed: {successes}, Duplicates: {duplicates}, Errors: {errors}")


# -----------------------------
# Search by Image
# -----------------------------
with tab_search:
    st.subheader("Search by Image")
    query_file = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        key="search_query_uploader",
    )
    col1, col2 = st.columns([1, 2])

    with col1:
        if query_file is not None:
            try:
                image_bytes = query_file.read()
                query_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                st.image(query_img, caption="Query Image", width="stretch")
                # Store the query image in session state to prevent re-upload issues
                st.session_state.query_image = query_img
            except Exception as e:
                query_img = None
                st.error(f"Invalid image: {e}")
        else:
            query_img = None

        top_k = st.number_input("Top K", min_value=1, max_value=50, value=top_k_setting, step=1)
        do_search = st.button("Search", disabled=(query_img is None) or (not qdrant_ok))

    with col2:
        if do_search and query_img is not None:
            if not qdrant_ok:
                st.error("Qdrant connection is not available. Please check settings.")
            else:
                with st.spinner("Searching similar images..."):
                    try:
                        results = search_similar(query_img, top_k=top_k)
                    except Exception as e:
                        results = []
                        st.error(f"Search failed: {e}")

                if results:
                    # Display in a responsive grid
                    cols_per_row = 5 if len(results) >= 5 else len(results)
                    rows = (len(results) + cols_per_row - 1) // cols_per_row
                    idx = 0
                    for _ in range(rows):
                        row_cols = st.columns(cols_per_row)
                        for c in row_cols:
                            if idx >= len(results):
                                break
                            path, score, payload = results[idx]
                            caption = f"{payload.get('filename','')}\nCosine: {score:.4f}"
                            if path and os.path.exists(path):
                                c.image(path, caption=caption, width="stretch")
                            else:
                                c.write(caption)
                                c.warning("Image file missing")
                            idx += 1
                else:
                    st.info("No results found.")


# -----------------------------
# Gallery & Manage
# -----------------------------
with tab_gallery:
    st.subheader("Gallery & Manage Images")
    
    # Get all images from the database
    if qdrant_ok:
        try:
            # Get all points from the collection
            points, _ = qdrant_client.scroll(collection_name=collection_name, limit=1000)
            
            if points:
                st.write(f"Found {len(points)} images in the database")
                
                # Create a grid layout for images
                cols_per_row = 4
                rows = (len(points) + cols_per_row - 1) // cols_per_row
                
                # Track selected images for deletion
                if 'selected_images' not in st.session_state:
                    st.session_state.selected_images = set()
                
                # Display images in grid
                for row in range(rows):
                    cols = st.columns(cols_per_row)
                    for col_idx in range(cols_per_row):
                        img_idx = row * cols_per_row + col_idx
                        if img_idx >= len(points):
                            break
                            
                        point = points[img_idx]
                        payload = point.payload or {}
                        filename = payload.get('filename', 'unknown')
                        filepath = payload.get('filepath', os.path.join(IMAGES_DIR, filename))
                        item_type = payload.get('type', 'unknown')
                        timestamp = payload.get('timestamp', 'unknown')
                        
                        with cols[col_idx]:
                            # Checkbox for selection
                            is_selected = st.checkbox(
                                f"Select {filename}",
                                key=f"select_{img_idx}",
                                value=img_idx in st.session_state.selected_images
                            )
                            
                            if is_selected:
                                st.session_state.selected_images.add(img_idx)
                            else:
                                st.session_state.selected_images.discard(img_idx)
                            
                            # Display image
                            if os.path.exists(filepath):
                                st.image(filepath, caption=filename, width="stretch")
                            else:
                                st.warning(f"File not found: {filename}")
                            
                            # Display metadata
                            st.caption(f"Type: {item_type}")
                            st.caption(f"Added: {timestamp[:10] if timestamp != 'unknown' else 'unknown'}")
                
                # Delete selected images
                if st.session_state.selected_images:
                    st.write(f"Selected {len(st.session_state.selected_images)} images for deletion")
                    
                    if st.button("Delete Selected Images", type="primary"):
                        deleted_count = 0
                        for img_idx in list(st.session_state.selected_images):
                            try:
                                point = points[img_idx]
                                payload = point.payload or {}
                                filename = payload.get('filename', 'unknown')
                                filepath = payload.get('filepath', os.path.join(IMAGES_DIR, filename))
                                
                                # Delete from Qdrant
                                qdrant_client.delete(
                                    collection_name=collection_name,
                                    points_selector=[point.id]
                                )
                                
                                # Delete local file
                                if os.path.exists(filepath):
                                    os.remove(filepath)
                                
                                deleted_count += 1
                                st.session_state.selected_images.discard(img_idx)
                                
                            except Exception as e:
                                st.error(f"Error deleting {filename}: {e}")
                        
                        st.success(f"Deleted {deleted_count} images")
                        st.rerun()
                
                # Clear selection button
                if st.session_state.selected_images:
                    if st.button("Clear Selection"):
                        st.session_state.selected_images.clear()
                        st.rerun()
                        
            else:
                st.info("No images found in the database. Upload some images first!")
                
        except Exception as e:
            st.error(f"Error loading gallery: {e}")
    else:
        st.error("Qdrant connection is not available. Please check settings.")


# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Built with DINOv2 ViT-B/14, Qdrant, and Streamlit. Embeddings normalized with L2.")


