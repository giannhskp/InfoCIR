import base64
import threading
import tempfile
from io import BytesIO
from dash import callback, Input, Output, State, html
import dash_bootstrap_components as dbc
from PIL import Image
from src import config
import os
import sys
# Ensure the SEARLE demo directory is on the path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SEARLE'))
from compose_image_retrieval_demo import ComposedImageRetrievalSystem
from src.Dataset import Dataset
import torch
import torch.nn.functional as F
import pickle

# Thread lock and CIR system instance
lock = threading.Lock()
cir_system = None

@callback(
    [Output('cir-upload-status', 'children'),
     Output('cir-query-preview', 'children'),
     Output('cir-search-button', 'disabled')],
    [Input('cir-upload-image', 'contents'),
     Input('cir-text-prompt', 'value')],
    [State('cir-upload-image', 'filename')]
)
def update_upload_status(upload_contents, text_prompt, filename):
    """Handle image upload and update UI components"""
    if upload_contents is None:
        return (html.Div("No image uploaded", className="text-muted small"), html.Div(), True)
    try:
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))
        preview = html.Div([
            html.H6("Query Image Preview:", className="mb-2"),
            dbc.Card(dbc.CardBody([
                html.Img(src=upload_contents, className="img-fluid", style={'max-width':'200px','max-height':'200px'}),
                html.P(f"Filename: {filename}", className="small text-muted mt-2 mb-0"),
                html.P(f"Size: {img.size[0]} x {img.size[1]}", className="small text-muted mb-0")
            ]), style={'width': 'fit-content'})
        ])
        uploaded = html.Div([html.I(className="fas fa-check-circle text-success me-2"), "Image uploaded successfully"], className="text-success small")
        disabled = not (upload_contents and text_prompt and text_prompt.strip())
        return uploaded, preview, disabled
    except Exception as e:
        error = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Error processing image: {e}"], className="text-danger small")
        return error, html.Div(), True

@callback(
    Output('cir-search-button', 'disabled', allow_duplicate=True),
    [Input('cir-text-prompt', 'value')],
    [State('cir-upload-image', 'contents')],
    prevent_initial_call=True
)
def update_search_button_state(text_prompt, upload_contents):
    """Enable/disable search button based on input validation"""
    if upload_contents and text_prompt and text_prompt.strip():
        return False
    return True

@callback(
    [Output('cir-results', 'children'),
     Output('cir-search-status', 'children'),
     Output('cir-search-results', 'data')],
    [Input('cir-search-button', 'n_clicks')],
    [State('cir-upload-image', 'contents'), State('cir-text-prompt', 'value'), State('cir-top-n', 'value')],
    prevent_initial_call=True
)
def perform_cir_search(n_clicks, upload_contents, text_prompt, top_n):
    """Perform CIR search using the SEARLE ComposedImageRetrievalSystem"""
    if not upload_contents or not text_prompt:
        empty = html.Div("No results yet. Upload an image and enter a text prompt to start retrieval.", className="text-muted text-center p-4")
        # Clear any previous CIR visualization
        return empty, html.Div(), None
    try:
        # Decode and save query image
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(decoded)
        tmp.close()

        global cir_system
        if cir_system is None:
            with lock:
                if cir_system is None:
                    cir_system = ComposedImageRetrievalSystem(
                        dataset_path=config.CIR_DATASET_PATH,
                        dataset_type=config.CIR_DATASET_TYPE,
                        clip_model_name=config.CIR_CLIP_MODEL_NAME,
                        eval_type=config.CIR_EVAL_TYPE,
                        preprocess_type=config.CIR_PREPROCESS_TYPE,
                        exp_name=config.CIR_EXP_NAME,
                        phi_checkpoint_name=config.CIR_PHI_CHECKPOINT_NAME
                    )
                    cir_system.create_database(split=config.CIR_SPLIT)

        with lock:
            results = cir_system.query(tmp.name, text_prompt, top_n)

        # Build result cards using paths from the loaded dataset
        cards = []
        df = Dataset.get()
        for img_name, score in results:
            img_path = None
            try:
                # Try numeric index lookup
                idx = int(img_name)
                if idx in df.index:
                    img_path = df.loc[idx]['image_path']
            except Exception:
                # Try string index lookup
                if img_name in df.index:
                    img_path = df.loc[img_name]['image_path']
            # Skip if path not found or doesn't exist
            if not img_path or not os.path.exists(img_path):
                continue
            data = open(img_path, 'rb').read()
            src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            card = dbc.Card(dbc.CardBody([
                html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
                html.P(f"Score: {score:.4f}", className='small text-center mt-1')
            ]), className='result-card')
            cards.append(card)

        rows = []
        for i in range(0, len(cards), 5):
            chunk = cards[i:i+5]
            cols = [dbc.Col(c, width=12//len(chunk)) for c in chunk]
            rows.append(dbc.Row(cols, className='mb-3'))

        results_div = html.Div([html.H5("Retrieved Images", className="mb-3")] + rows)
        status = html.Div([html.I(className="fas fa-check-circle text-success me-2"), f"Retrieved {len(cards)} images"], className="text-success small")
        # Prepare store data for visualization
        # Map retrieved image names to DataFrame indices
        topk_ids = []
        for img_name, _ in results:
            try:
                idx = int(img_name)
                if idx in df.index:
                    topk_ids.append(idx)
            except:
                if img_name in df.index:
                    topk_ids.append(img_name)
        top1_id = topk_ids[0] if topk_ids else None
        # Compute query embedding and normalize
        device_model = next(cir_system.clip_model.parameters()).device
        query_img = Image.open(tmp.name).convert('RGB')
        query_input = cir_system.preprocess(query_img).unsqueeze(0).to(device_model)
        with torch.no_grad():
            feat = cir_system.clip_model.encode_image(query_input)
            feat = F.normalize(feat.float(), dim=-1)
        feat_np = feat.cpu().numpy()
        # UMAP transform
        umap_path = config.WORK_DIR / 'umap_reducer.pkl'
        umap_reducer = pickle.load(open(str(umap_path), 'rb'))
        umap_xy = umap_reducer.transform(feat_np)
        umap_x_query, umap_y_query = float(umap_xy[0][0]), float(umap_xy[0][1])
        # TSNE approximate by nearest neighbor
        db_feats = cir_system.database_features.cpu()
        with torch.no_grad():
            sims = (feat.cpu() @ db_feats.T).squeeze(0)
        idx_nn = int(torch.argmax(sims).item())
        nn_name = cir_system.database_names[idx_nn]
        try:
            nn_id = int(nn_name)
        except:
            nn_id = nn_name
        tsne_x_query = float(df.loc[nn_id]['tsne_x'])
        tsne_y_query = float(df.loc[nn_id]['tsne_y'])
        # Delete temporary query image file
        os.unlink(tmp.name)
        store_data = {
            'topk_ids': topk_ids,
            'top1_id': top1_id,
            'umap_x_query': umap_x_query,
            'umap_y_query': umap_y_query,
            'tsne_x_query': tsne_x_query,
            'tsne_y_query': tsne_y_query
        }
        return results_div, status, store_data
    except Exception as e:
        err = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Retrieval error: {e}"], className="text-danger small")
        return html.Div("Error occurred during image retrieval.", className="text-danger text-center p-4"), err, None