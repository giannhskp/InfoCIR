import base64
import threading
import tempfile
from io import BytesIO
from dash import callback, Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from PIL import Image
from src import config
import os
import sys
from src.Dataset import Dataset
import torch
import torch.nn.functional as F
import pickle
import clip
from dash.exceptions import PreventUpdate
from dash import ALL
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from dash import dcc
import plotly.graph_objects as go
from src.widgets import gallery, wordcloud, histogram, scatterplot
from src.callbacks.saliency_callbacks import load_and_resize_image  # Reuse efficient thumbnail loader
from dash import no_update
import copy
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional

@callback(
    [Output('cir-upload-status', 'children'),
     Output('cir-upload-preview', 'children'),
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
            html.Div([
                html.Img(src=upload_contents,
                         style={'width':'40px','height':'40px','objectFit':'cover','borderRadius':'4px','border':'1px solid #ccc'}),
                html.Span(f" {filename}", className="small text-muted ms-2")
            ], className='d-flex align-items-center'),
        ], style={'maxHeight':'50px','overflow':'hidden'})
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
     Output('cir-search-data', 'data'),
     Output('cir-toggle-button', 'style', allow_duplicate=True),
     Output('cir-toggle-button', 'disabled', allow_duplicate=True),
     Output('cir-toggle-button', 'children', allow_duplicate=True),
     Output('cir-toggle-button', 'color', allow_duplicate=True),
     Output('cir-toggle-state', 'data', allow_duplicate=True),
     Output('cir-run-button', 'style'),
     Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data', allow_duplicate=True),
     Output('viz-mode', 'data', allow_duplicate=True),
     Output('cir-selected-image-ids', 'data', allow_duplicate=True),
     Output('viz-selected-ids', 'data', allow_duplicate=True),
     Output('saliency-data', 'data'),
     Output('wordcloud', 'list', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True)],
    [Input('cir-search-button', 'n_clicks')],
    [State('cir-upload-image', 'contents'),
     State('cir-text-prompt', 'value'),
     State('cir-top-n', 'value'),
     State('cir-toggle-state', 'data')],
    prevent_initial_call=True
)
def perform_cir_search(n_clicks, upload_contents, text_prompt, top_n, current_toggle_state):
    """Perform CIR search using the SEARLE ComposedImageRetrievalSystem"""
    top_n = int(top_n)
    selected_model = "SEARLE"  # Hardcoded model selection
    if not upload_contents or not text_prompt:
        from src.widgets import histogram
        empty = html.Div("No results yet. Upload an image and enter a text prompt to start retrieval.", className="text-muted text-center p-4")
        # No search yet → keep Visualize button disabled, ensure Run-CIR hidden.
        return (
            empty,                 # cir-results
            html.Div(),            # cir-search-status
            None,                  # cir-search-data
            no_update,             # cir-toggle-button style (unchanged)
            True,                  # cir-toggle-button disabled
            'Visualize CIR results',  # cir-toggle-button text
            'success',             # cir-toggle-button color
            False,                 # cir-toggle-state OFF
            {'display': 'none'},   # cir-run-button style
            [],                    # cir-enhance-results
            None,                  # cir-enhanced-prompts-data
            False,                 # viz-mode
            [],                    # cir-selected-image-ids
            [],                    # viz-selected-ids
            None,                  # saliency-data
            [],                    # wordcloud
            histogram.draw_histogram(None)  # histogram figure
        )
    try:
        print(f"Starting CIR search with model: {selected_model}, prompt: '{text_prompt}', top_n: {top_n}")
        
        # Visualization mode is always OFF when a new search starts. Define it
        # here so that subsequent card-building code can reference it safely.
        viz_mode = False

        # Decode and save query image
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(decoded)
        tmp.close()
        print(f"Query image saved to: {tmp.name}")

        print(f"Selected CIR model: {selected_model}")

        # Use saliency-enabled CIR query
        from src.saliency import perform_cir_with_saliency, get_saliency_status_message
        print("Calling perform_cir_with_saliency...")
        results, saliency_data = perform_cir_with_saliency(
            temp_image_path=tmp.name,
            text_prompt=text_prompt,
            top_n=top_n,
            selected_model=selected_model
        )
        print(f"CIR search completed. Got {len(results)} results")

        # Build result cards using paths from the loaded dataset
        cards = []
        df = Dataset.get()
        print(f"Dataset has {len(df)} entries")
        
        for card_index, (img_name, score) in enumerate(results):
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
                print(f"Warning: Could not find image path for {img_name}")
                continue
                
            # Load and resize image efficiently (thumbnail ~150px, cached)
            try:
                src = load_and_resize_image(img_path, max_width=150, max_height=150)
                if not src:
                    print(f"Error: thumbnail generation returned None for {img_path}")
                    continue
            except Exception as e:
                print(f"Error thumbnailing {img_path}: {e}")
                continue
            
            # Create card body with improved styling
            card_body = dbc.CardBody([
                html.Img(
                    src=src,
                    className="img-fluid",
                    style={
                        "maxWidth": "100%",
                        "height": "auto",
                        "objectFit": "contain",
                        "borderRadius": "4px",
                        "marginBottom": "8px",
                    },
                ),
                html.Div([
                    html.Small(
                        f"{score:.3f}",
                        className="badge bg-primary",
                        style={"fontSize": "0.65rem", "fontWeight": "500"},
                    )
                ], className="text-center"),
            ], style={"padding": "8px", "display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"})
            
            # TOP-1 card behaviour differs between modes
            if card_index == 0:
                inner = dbc.Card(card_body, className="result-card", style={"border": "1px solid #dee2e6", "borderRadius": "6px", "height": "100%"})

                if viz_mode:
                    # Clickable wrapper (multi-select in visualization)
                    wrapper = html.Div([
                        inner,
                        html.Div("TOP-1", className="badge bg-warning position-absolute top-0 start-0 m-1"),
                    ], id={"type": "cir-result-card", "index": str(img_name)}, n_clicks=0,
                    className="result-card-wrapper position-relative", style={"height": "100%"})
                else:
                    # Non-clickable in prompt-enhancement mode
                    wrapper = html.Div([
                        inner,
                        html.Div("TOP-1", className="badge bg-warning position-absolute top-0 start-0 m-1"),
                    ], className="result-card-wrapper position-relative", style={"cursor": "default", "height": "100%"})
            else:
                # Other images are clickable
                inner_card = dbc.Card(
                    card_body, 
                    className='result-card',
                    style={
                        'border': '1px solid #dee2e6',
                        'borderRadius': '6px',
                        'height': '100%'
                    }
                )
                # Highlight if this image was already selected before rebuilding
                card = html.Div(
                    inner_card,
                    id={'type': 'cir-result-card', 'index': img_name},
                    n_clicks=0,
                    className="result-card-wrapper",
                    style={'height': '100%'}
                )
            if card_index == 0:
                cards.append(wrapper)
            else:
                cards.append(card)

        print(f"Created {len(cards)} result cards")

        rows = []
        for i in range(0, len(cards), 4):  # Changed from 5 to 4 cards per row for better spacing
            chunk = cards[i:i+4]
            cols = [dbc.Col(c, width=3, className='mb-3 px-2') for c in chunk]  # Fixed width and added padding
            rows.append(dbc.Row(cols, className='g-2'))

        header = html.Div([
            html.H5("Retrieved Images", className="mb-0", style={"display": "inline-block"}),
            dbc.Button(id="visualize-toggle-button", size="sm", color="secondary", class_name="ms-2", n_clicks=0,
                       children="Visualize OFF")
        ], className="d-flex align-items-center mb-3")

        results_div = html.Div([header] + rows)
        
        # Create status message with saliency information
        status_messages = [html.I(className="fas fa-check-circle text-success me-2"), f"Retrieved {len(cards)} images"]
        saliency_status = get_saliency_status_message(saliency_data)
        if saliency_status:
            status_messages.extend([html.Br(), html.I(className="fas fa-brain text-info me-2"), saliency_status])
        
        status = html.Div(status_messages, className="text-success small")
        
        # Reduce saliency data to a lightweight summary (only save_directory) to avoid large JSON payloads
        saliency_summary = None
        if saliency_data and isinstance(saliency_data, dict):
            save_dir = saliency_data.get('save_directory')
            if save_dir:
                saliency_summary = {'save_directory': save_dir}

                # Include token attribution data for in-app visualisation (lightweight)
                if 'text_attribution' in saliency_data:
                    saliency_summary['text_attribution'] = saliency_data['text_attribution']
        
        # Prepare store data for visualization - simplified version
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
        
        # ---------------------------------------------------------
        # Compute UMAP coordinates for the Query and Final Query
        # (very small payload: just 4 floats) so they can be plotted
        # ---------------------------------------------------------
        try:
            from src.shared import cir_systems  # Local import to avoid circulars at top level
            device_model = next(cir_systems.cir_system_searle.clip_model.parameters()).device
            from PIL import Image as PILImage
            query_img = PILImage.open(tmp.name).convert('RGB')
            query_input = cir_systems.cir_system_searle.preprocess(query_img).unsqueeze(0).to(device_model)
            # ------------------------------------------------------------------
            # Obtain CLIP image embedding **without** normalisation for φ.
            # We keep a *separate* L2-normalised copy for downstream projection
            # (UMAP) but pass the raw vector to the pseudo-token network, as
            # required by SEARLE (normalising beforehand distorts the feature
            # distribution that φ was trained on).
            # ------------------------------------------------------------------
            with torch.no_grad():
                img_feat_raw = cir_systems.cir_system_searle.clip_model.encode_image(query_input).float()
                img_feat = F.normalize(img_feat_raw, dim=-1)
            feat_np = img_feat.cpu().numpy()

            # Compute final composed query embedding only for SEARLE models that have φ network
            final_query_feat_np = None
            if getattr(cir_systems.cir_system_searle, 'phi', None) is not None:
                try:
                    with torch.no_grad():
                        pseudo_tokens = cir_systems.cir_system_searle.phi(img_feat_raw)
                        input_caption = f"a photo of $ that {text_prompt}"
                        tokenized_caption = clip.tokenize([input_caption], context_length=77).to(device_model)
                        from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
                        final_q_feat = encode_with_pseudo_tokens(
                            cir_systems.cir_system_searle.clip_model,
                            tokenized_caption,
                            pseudo_tokens
                        ).float()
                        # Normalise along the embedding dimension (-1)
                        final_q_feat = F.normalize(final_q_feat, dim=-1)
                        # Store as NumPy float32 to avoid dtype issues inside
                        # scikit-learn transformers.
                        final_query_feat_np = final_q_feat.cpu().numpy().astype(np.float32)
                except Exception as e:
                    print(f"Warning: failed to compute final query features: {e}")

            # ------------------------------------------------------------------
            # Project Query (+ Final Query) to 2-D UMAP space using the same PCA
            # preprocessing that was fitted during dataset projection.
            # ------------------------------------------------------------------

            umap_path = config.WORK_DIR / 'umap_reducer.pkl'
            pipeline_path = config.WORK_DIR / 'projection_pipeline.pkl'

            def _apply_projection_pipeline(arr_np):
                """Apply the saved pre-UMAP projection pipeline (style debias →
                contrastive debias → alternative proj → final PCA) to *arr_np*
                (shape: N×D).  If the pipeline file is missing we fall back to
                the raw vectors so the app keeps working."""

                if not os.path.exists(pipeline_path):
                    return arr_np  # no pipeline – raw features

                try:
                    pipe = pickle.load(open(str(pipeline_path), 'rb'))

                    x = arr_np.copy()

                    # --- Style debiasing (scaler → PCA → semantic dims) ---
                    if pipe.get('style_scaler') is not None:
                        x = pipe['style_scaler'].transform(x)
                        x = pipe['style_pca'].transform(x)
                        x = x[:, pipe['style_dims']]

                    # --- Contrastive debiasing (approximate: nearest prototype) ---
                    if pipe.get('contrastive_scaler') is not None:
                        x_scaled = pipe['contrastive_scaler'].transform(x)
                        protos = pipe['contrastive_prototypes']
                        weight = pipe['contrastive_weight']
                        # Find nearest prototype in L2 sense
                        dists = ((protos - x_scaled)**2).sum(axis=1)
                        nearest_idx = int(dists.argmin())
                        proto_vec = protos[nearest_idx]
                        x_scaled = x_scaled + weight * (proto_vec - x_scaled)
                        x = x_scaled  # remain in scaled space

                    # --- Alternative projection ---
                    if pipe.get('alt_scaler') is not None:
                        x = pipe['alt_scaler'].transform(x)
                        alt_model = pipe['alt_model']
                        x = alt_model.transform(x)

                    # --- Final PCA ---
                    if pipe.get('final_pca') is not None:
                        x = pipe['final_pca'].transform(x)

                    return x
                except Exception as e:
                    print(f"Warning: failed to apply projection pipeline: {e}")
                    return arr_np

            if os.path.exists(umap_path):
                umap_reducer = pickle.load(open(str(umap_path), 'rb'))

                # --- Query ---
                proj_input = _apply_projection_pipeline(feat_np)
                umap_xy = umap_reducer.transform(proj_input)
                umap_x_query, umap_y_query = float(umap_xy[0][0]), float(umap_xy[0][1])

                # --- Final Query ---
                if final_query_feat_np is not None:
                    proj_final = _apply_projection_pipeline(final_query_feat_np)
                    final_umap_xy = umap_reducer.transform(proj_final)
                    umap_x_final_query, umap_y_final_query = float(final_umap_xy[0][0]), float(final_umap_xy[0][1])
                    xfq, yfq = umap_x_final_query, umap_y_final_query
                else:
                    umap_x_final_query = umap_y_final_query = None
            else:
                umap_x_query = umap_y_query = umap_x_final_query = umap_y_final_query = None
        except Exception as e:
            print(f"Warning: failed to compute query UMAP coords: {e}")
            umap_x_query = umap_y_query = umap_x_final_query = umap_y_final_query = None

        # Delete temporary query image file
        os.unlink(tmp.name)
        print("Temporary query image deleted")
        
        # Simplified store data - we'll compute embeddings later if needed for visualization
        store_data = {
            'topk_ids': topk_ids,
            'top1_id': top1_id,
            'umap_x_query': umap_x_query,
            'umap_y_query': umap_y_query,
            'umap_x_final_query': umap_x_final_query,
            'umap_y_final_query': umap_y_final_query,

            'text_prompt': text_prompt,
            'top_n': top_n,
            'upload_contents': upload_contents,
            # 🔒 Persist original retrieval results so that we can rebuild the cards later when
            #    the user toggles between the baseline query and enhanced prompts.
            'original_results': [[str(name), float(score)] for (name, score) in results],
        }
        
        # Calculate histogram and wordcloud based on current visualization state
        from src.widgets import wordcloud, histogram
        if current_toggle_state:  # If visualization is currently enabled
            # Populate histogram and wordcloud with CIR results
            counts = df.loc[topk_ids]['class_name'].value_counts()
            if len(counts):
                wg = wordcloud.wordcloud_weight_rescale(counts.values, 1, counts.max())
                wc = sorted([[cn, w] for cn, w in zip(counts.index, wg)], key=lambda x: x[1], reverse=True)
            else:
                wc = []
            
            # Create histogram for CIR results
            cir_df = df.loc[topk_ids]
            hist = histogram.draw_histogram(cir_df)
        else:
            # Visualization is OFF - return empty widgets
            wc = []  # Empty wordcloud when visualization is OFF
            hist = histogram.draw_histogram(None)  # Empty histogram when visualization is OFF
        
        print("CIR search callback completed successfully")
        # Auto-enable Visualize button (set ON) and keep Run-CIR hidden, reset viz-related stores
        return (
            results_div,               # cir-results
            status,                    # cir-search-status
            store_data,                # cir-search-data
            no_update,                 # cir-toggle-button style (unchanged)
            False,                     # cir-toggle-button disabled – enabled for interaction
            'Hide CIR results',        # cir-toggle-button text (now ON)
            'warning',                 # cir-toggle-button color
            True,                      # cir-toggle-state – ON
            {'display': 'none'},     # cir-run-button style – hidden
            [],                        # cir-enhance-results
            None,                      # cir-enhanced-prompts-data
            False,                     # viz-mode – OFF by default
            [],                        # cir-selected-image-ids
            [],                        # viz-selected-ids
            saliency_summary,          # saliency-data
            wc,                        # wordcloud
            hist                       # histogram figure
        )
    except Exception as e:
        print(f"CIR search error: {e}")
        import traceback
        traceback.print_exc()
        err = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Retrieval error: {e}"], className="text-danger small")
        # On error, keep Visualize button disabled and Run-CIR hidden
        return (
            html.Div("Error occurred during image retrieval.", className="text-danger text-center p-4"),  # cir-results
            err,                         # cir-search-status
            None,                        # cir-search-data
            no_update,                   # cir-toggle-button style
            True,                        # cir-toggle-button disabled
            'Visualize CIR results',     # button text
            'success',                   # button color
            False,                       # cir-toggle-state OFF
            {'display': 'none'},        # cir-run-button style – hidden
            [],                          # cir-enhance-results
            None,                        # cir-enhanced-prompts-data
            False,                       # viz-mode
            [],                          # cir-selected-image-ids
            [],                          # viz-selected-ids
            None,                        # saliency-data
            [],                          # wordcloud
            histogram.draw_histogram(None)  # histogram figure
        )

# Button toggle callback for CIR visualization
@callback(
    [Output('cir-toggle-button', 'children'),
     Output('cir-toggle-button', 'color'),
     Output('cir-toggle-button', 'style', allow_duplicate=True),
     Output('cir-toggle-state', 'data')],
    Input('cir-toggle-button', 'n_clicks'),
    State('cir-toggle-state', 'data'),
    prevent_initial_call=True
)
def toggle_cir_visualization(n_clicks, current_state):
    """Toggle between visualize and hide CIR results"""
    if n_clicks is None:
        return 'Visualize CIR results', 'success', {'display': 'block', 'color': 'black'}, False
    
    # Toggle state
    new_state = not current_state
    
    if new_state:  # Now visualizing
        return 'Hide CIR results', 'warning', {'display': 'block', 'color': 'black'}, True
    else:  # Now hidden
        return 'Visualize CIR results', 'success', {'display': 'block', 'color': 'black'}, False

# Disabled dynamic enhance button creation; static button is in layout

# New callback to update the 'Enhance prompt' button based on selection
@callback(
    [Output('enhance-prompt-button', 'disabled'),
     Output('enhance-prompt-button', 'color')],
    [Input({'type': 'cir-result-card', 'index': ALL}, 'className'),
     Input('cir-selected-image-ids', 'data')]
)
def update_enhance_button_state(wrapper_classnames, selected_ids):
    """
    Enable the Enhance prompt button when a result is selected; otherwise keep it disabled.
    """
    # Check both className and selected_ids to handle clearing from mode toggle
    has_selected_class = any('selected' in cn for cn in wrapper_classnames)
    has_selected_ids = selected_ids and len(selected_ids) > 0
    
    if has_selected_class or has_selected_ids:
        return False, 'primary'
    return True, 'secondary'

@callback(
    [Output({'type': 'cir-result-card', 'index': ALL}, 'className'),
     Output('cir-selected-image-ids', 'data')],
    Input({'type': 'cir-result-card', 'index': ALL}, 'n_clicks'),
    [State({'type': 'cir-result-card', 'index': ALL}, 'className'),
     State('viz-mode', 'data'),
     State('cir-selected-image-ids', 'data')],
    prevent_initial_call=True
)
def toggle_cir_result_selection(n_clicks_list, current_classnames, viz_mode, selected_ids):
    """
    Toggle selection highlight for CIR result cards, allowing MULTIPLE selections.
    Clicking a selected card again will deselect it.
    """
    # If visualization mode is ON, ignore prompt-enhancement selection logic
    if viz_mode:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # --------------------------------------------------------------
    # Ignore the callback invocation that happens when the layout is
    # first rendered. During that moment Dash sets n_clicks = 0 for
    # every result card which would otherwise be interpreted here as
    # a *click*. We only care about genuine user clicks where the
    # clicked card's n_clicks > 0.
    # --------------------------------------------------------------
    triggered_id_raw = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        trig_dict = json.loads(triggered_id_raw)
        clicked_idx = str(trig_dict.get('index'))
    except Exception:
        raise PreventUpdate

    # Find the n_clicks value corresponding to the triggered card
    clicked_n_clicks = None
    for inp_dict, n_val in zip(ctx.inputs_list[0], n_clicks_list):
        if str(inp_dict['id']['index']) == clicked_idx:
            clicked_n_clicks = n_val
            break

    if clicked_n_clicks is None or clicked_n_clicks == 0:
        # Layout-initialisation trigger – ignore
        raise PreventUpdate

    # Ensure list initialised
    selected_ids = selected_ids or []

    # Toggle membership
    if clicked_idx in selected_ids:
        selected_ids.remove(clicked_idx)
        was_selected = True
    else:
        selected_ids.append(clicked_idx)
        was_selected = False

    # ------------------------------------------------------------------
    # Build updated class names list (add/remove 'selected')
    # ------------------------------------------------------------------
    new_classnames = []
    sel_set = set(selected_ids)
    for input_dict, cls in zip(ctx.inputs_list[0], current_classnames):
        idx = str(input_dict['id']['index'])
        parts = cls.split()
        if idx in sel_set and 'selected' not in parts:
            parts.append('selected')
        if idx not in sel_set and 'selected' in parts:
            parts.remove('selected')
        new_classnames.append(' '.join(parts))

    return new_classnames, selected_ids

# New callback to enhance the user prompt and evaluate against the selected image
@callback(
    [Output('cir-search-status', 'children', allow_duplicate=True),
     Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data')],
    Input('enhance-prompt-button', 'n_clicks'),
    [State('cir-search-data', 'data'), State('cir-selected-image-ids', 'data'), State('saliency-data', 'data')],
    prevent_initial_call=True
)
def enhance_prompt(n_clicks, search_data, selected_image_ids, saliency_summary):
    """
    Enhance the user's prompt via a small LLM, compare each to the selected image, choose the best,
    rerun CIR with that prompt, and display diagnostics.
    """
    print(f"Performing prompt enhancement for {len(selected_image_ids)} selected images")
    print(f"Selected image IDs: {selected_image_ids}")
    
    import os
    # Guard against missing data
    if not search_data or not selected_image_ids:
        raise PreventUpdate

    # Reconstruct query image file
    _, content_string = search_data['upload_contents'].split(',')
    decoded = base64.b64decode(content_string)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tmp.write(decoded)
    tmp.close()

    # Extract class and style information from selected ideal images (if enabled)
    context_info = ""
    if config.ENHANCEMENT_USE_CONTEXT:
        class_style_info = extract_class_and_style_info(selected_image_ids)
        classes = class_style_info['classes']
        styles = class_style_info['styles']
        
        # Build context information for the LLM
        if classes:
            context_info += f"The user has selected ideal images from these classes: {', '.join(classes)}. "
        if styles:
            context_info += f"The selected images represent these artistic styles: {', '.join(styles)}. "
    
    # Prepare LLM for prompt enhancement using Mistral-7B-Instruct
    N = config.ENHANCEMENT_CANDIDATE_PROMPTS  # Number of candidate prompts to generate
    original_prompt = search_data['text_prompt']
    MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    print(f"Loading enhancement model {MODEL_NAME} on {DEVICE}...")  # Debug log
    print(f"Context enhancement enabled: {config.ENHANCEMENT_USE_CONTEXT}")  # Debug log
    if config.ENHANCEMENT_USE_CONTEXT:
        print(f"Context info: {context_info}")  # Debug log
    os.environ['HF_TOKEN'] = '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'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    
    # Enhanced instruction with class and style context
    base_instruction = f"""
    You are an assistant helping improve short prompts for image retrieval. 
    Given a query like: "{original_prompt}", generate one short, reworded version 
    that retains the original meaning but adds slight variety or detail. 
    Do NOT describe scenes or characters. Just rephrase the original style-focused prompt.
    """
    
    if context_info:
        instruction = f"""
        {base_instruction}
        
        Additional context: {context_info}Use this information to make the enhanced prompt more relevant to these classes and styles while maintaining the original intent. Do not specifically mention any of the classes in the enhanced prompt.

        Original prompt: "{original_prompt}"

        Return only one short enhanced prompt enclosed inside <ANSWER> </ANSWER> tags.
        """
    else:
        instruction = f"""
        {base_instruction}

        Original prompt: "{original_prompt}"

        Return only one short enhanced prompt enclosed inside <ANSWER> </ANSWER> tags.
        """

    formatted_prompt = f"<s>[INST] {instruction.strip()} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=1.2,
        top_p=0.8,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=N,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Enhancement results: {len(results)}")

    # Extract enhanced prompts from between <ANSWER> tags (case-insensitive)
    prompts = []
    for i, result in enumerate(results):
        result = result.lower()
        # First extract only the response part after [/INST]
        inst_idx = result.find('[/inst]')
        if inst_idx == -1:
            print(f"No [/INST] found in result {i+1}")
            continue
        
        response_part = result[inst_idx + 7:]  # Skip '[/INST]'
        
        # Now look for ANSWER tags in the response part only
        start_tag = '<answer>'
        end_tag = '</answer>'
        start_idx = response_part.find(start_tag)
        end_idx = response_part.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            prompt = response_part[start_idx + len(start_tag):end_idx].strip()
            if prompt:  # Only add non-empty prompts
                prompts.append(prompt)
            else:
                print(f"Prompt was empty after stripping for result {i+1}")
    
    # If no valid prompts found, use original as fallback
    if not prompts:
        print("No prompts extracted, using original as fallback")
        prompts = [original_prompt]
    
    # clean up prompts
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if p]
    # remove '"' from prompts
    prompts = [p.replace('"', '') for p in prompts]
    # Remove duplicates
    prompts = list(set(prompts))
    print(f"Final prompts list: {prompts}")

    # ----------------------------------------
    # Helper metric functions (IR measures)
    # ----------------------------------------
    top_k = search_data['top_n']  # evaluation depth

    def calculate_ndcg(ranks):
        """Binary nDCG@k given 1-based ranks of the |selected_image_ids| relevant items
        (using the cut-off k = top_k)."""
        dcg = 0.0
        for r in ranks:
            if r <= top_k:
                dcg += 1.0 / math.log2(r + 1)
        m = len(selected_image_ids)
        ideal_dcg = sum(1.0 / math.log2(i + 1 + 1) for i in range(min(m, top_k)))
        return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    def calculate_average_precision(ranks):
        """Average-Precision@k (MAP for single query) – binary relevance."""
        num_rel_seen = 0
        precisions = []
        for r in sorted(ranks):
            if r > top_k:
                continue
            num_rel_seen += 1
            precisions.append(num_rel_seen / r)
        m = min(len(selected_image_ids), top_k)
        return sum(precisions) / m if m > 0 else 0.0

    def calculate_mrr(ranks):
        """Reciprocal rank of the first relevant result (0 if none in top-k)."""
        rr = min([r for r in ranks if r <= top_k], default=None)
        return 1.0 / rr if rr is not None else 0.0

    # Score each candidate prompt and store full results
    coverages = []       # Fraction of ideal images retrieved within top-k (0–1)
    mean_ranks = []      # Average rank of the selected ideal images (lower is better)
    mean_sims = []       # Mean similarity of the selected ideal images (for additional insight)
    ndcgs      = []      # Normalised Discounted Cumulative Gain@k (0–1)
    aps        = []      # Average-Precision@k  (0–1)
    mrrs       = []      # Mean Reciprocal Rank (0–1)
    all_prompt_results = []
    
    # Use saliency-enabled enhanced prompt processing
    from src.saliency import perform_enhanced_prompt_cir_with_saliency
    # Determine base directory for enhanced prompt saliency (inside initial query's directory if available)
    base_saliency_dir = None
    if isinstance(saliency_summary, dict):
        base_saliency_dir = saliency_summary.get('save_directory')

    all_prompt_results, enhanced_saliency_data = perform_enhanced_prompt_cir_with_saliency(
        temp_image_path=tmp.name,
        enhanced_prompts=prompts,
        top_n=search_data['top_n'],
        selected_image_ids=selected_image_ids,
        base_save_dir=base_saliency_dir
    )
    
    # Process results for scoring (multiple ideal images)
    for i, full_prompt_results in enumerate(all_prompt_results):
        # Build lookup for quick rank & sim
        name_to_rank = {}
        name_to_sim = {}
        for idx, (name, score) in enumerate(full_prompt_results):
            name_to_rank[str(name)] = idx + 1  # 1-based
            name_to_sim[str(name)] = score

        ranks_for_prompt = []
        sims_for_prompt = []
        for iid in selected_image_ids:
            if iid in name_to_rank:
                ranks_for_prompt.append(name_to_rank[iid])
                sims_for_prompt.append(name_to_sim[iid])
            else:
                # Not retrieved within top-k
                ranks_for_prompt.append(len(full_prompt_results) + 1)
                sims_for_prompt.append(0.0)

        retrieved_cnt = sum(r <= len(full_prompt_results) for r in ranks_for_prompt)
        coverage = retrieved_cnt / len(selected_image_ids)
        mean_rank = sum(ranks_for_prompt) / len(ranks_for_prompt)
        mean_sim = sum(sims_for_prompt) / len(sims_for_prompt)
        ndcg = calculate_ndcg(ranks_for_prompt)
        ap = calculate_average_precision(ranks_for_prompt)
        mrr = calculate_mrr(ranks_for_prompt)

        coverages.append(coverage)
        mean_ranks.append(mean_rank)
        mean_sims.append(mean_sim)
        ndcgs.append(ndcg)
        aps.append(ap)
        mrrs.append(mrr)

    # Select best prompt – optimise across metrics: nDCG ➜ AP ➜ Coverage ➜ Mean-rank
    best_idx = min(
        range(len(prompts)),
        key=lambda i: (
            -ndcgs[i],   # higher better
            -aps[i],
            -coverages[i],
            mean_ranks[i]
        )
    )
    best_prompt = prompts[best_idx]
    best_coverage = coverages[best_idx]
    best_mean_rank = mean_ranks[best_idx]
    best_ndcg = ndcgs[best_idx]
    best_ap = aps[best_idx]
    print(f"Selected best prompt: '{best_prompt}'  nDCG:{best_ndcg:.3f}  AP:{best_ap:.3f}  Coverage:{best_coverage:.2f}")

    # Get results for best prompt (already computed)
    full_results = all_prompt_results[best_idx]
    
    # Clean up temporary file
    os.unlink(tmp.name)

    # Status message with icon
    status_messages = [
        html.I(className="fas fa-magic text-success me-2"),
        "Enhanced prompt generated successfully! See analysis on the right column."
    ]
    
    # Add saliency status if available
    if enhanced_saliency_data:
        from src.saliency import get_saliency_status_message
        enhanced_saliency_status = get_saliency_status_message(enhanced_saliency_data)
        if enhanced_saliency_status:
            status_messages.extend([html.Br(), html.I(className="fas fa-brain text-info me-2"), enhanced_saliency_status])
    
    status = html.Div(status_messages, className="text-success small mb-3")

    # Create enhanced table rows with highlighting for best prompt and action buttons
    table_rows = []
    for i in range(len(prompts)):
        p = prompts[i]
        cov = coverages[i]
        ndcg_val = ndcgs[i]
        ap_val = aps[i]
        view_button = dbc.Button(
            [html.I(className="fas fa-eye me-1"), "View"],
            id={'type': 'enhanced-prompt-view', 'index': i},
            size="sm",
            color="outline-primary" if i != best_idx else "primary",
            className="btn-sm"
        )
        
        if i == best_idx:  # Highlight best prompt
            row = html.Tr([
                html.Td([html.I(className="fas fa-crown text-warning me-2"), p], className="fw-bold"),
                html.Td([html.Span(f"{ndcg_val*100:.0f}%", className="badge bg-success")]),
                html.Td([html.Span(f"{ap_val*100:.0f}%", className="badge bg-success")]),
                html.Td([html.Span(f"{int(cov*100)}%", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(f"{ndcg_val*100:.0f}%", className="badge bg-secondary")),
                html.Td(html.Span(f"{ap_val*100:.0f}%", className="badge bg-secondary")),
                html.Td(html.Span(f"{int(cov*100)}%", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with better styling and action column – columns: Coverage, Mean Rank
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
            html.Th([html.I(className="fas fa-chart-line me-2"), "nDCG"], className="bg-light"),
            html.Th([html.I(className="fas fa-percentage me-2"), "AP"], className="bg-light"),
            html.Th([html.I(className="fas fa-bullseye me-2"), "Coverage"], className="bg-light"),
            html.Th([html.I(className="fas fa-cogs me-2"), "Actions"], className="bg-light")
        ]), className="thead-light"),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, responsive=True, className="mb-4 shadow-sm")

    # Build image cards for CIR results of best prompt
    df = Dataset.get()
    cards = []
    for img_name, score in full_results:
        img_path = None
        try:
            idx = int(img_name)
            if idx in df.index:
                img_path = df.loc[idx]['image_path']
        except:
            if img_name in df.index:
                img_path = df.loc[img_name]['image_path']
        if not img_path or not os.path.exists(img_path):
            continue
        data = open(img_path, "rb").read()
        src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        # Create card body with score (like original CIR)
        card_body = dbc.CardBody([
            html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
            html.P(f"Score: {score:.4f}", className='small text-center mt-1')
        ])
        card = dbc.Card(card_body, className='result-card', style={'cursor': 'default'})
        cards.append(dbc.Col(card, width=2))

    # Arrange cards into rows of up to 6
    rows = []
    for i in range(0, len(cards), 6):
        rows.append(dbc.Row(cards[i:i+6], className="g-2 mb-3"))
    images_grid = html.Div(rows)

    # Build enhance results children with improved styling
    enhance_children = [
        # Main title with icon
        html.Div([
            html.I(className="fas fa-brain text-primary me-2"),
            html.H5("Enhanced Prompt Analysis", className="d-inline text-primary")
        ], className="mt-3 mb-4"),
        
        # Candidates section
        html.Div([
            html.H6([html.I(className="fas fa-list-alt me-2"), "Candidate Prompts & Scores"], className="mb-3 text-secondary"),
            candidates_table
        ]),
        
        # Best prompt highlight card
        dbc.Alert([
            html.H6([html.I(className="fas fa-trophy text-warning me-2"), "Selected Best Prompt"], className="alert-heading mb-2"),
            html.P(f'"{best_prompt}"', className="mb-1 font-monospace"),
            html.Small(f"nDCG: {best_ndcg*100:.0f}% | AP: {best_ap*100:.0f}% | Coverage: {best_coverage*100:.0f}%", className="text-muted")
        ], color="light", className="border-start border-warning border-4"),
        
        # Results section
        html.Div([
            html.H6([html.I(className="fas fa-images me-2"), "CIR Results for Best Prompt"], className="mt-4 mb-3 text-secondary"),
            images_grid
        ])
    ]

    # Build saliency directory list aligned with prompts
    prompt_saliency_dirs = []
    if enhanced_saliency_data and 'prompt_saliency' in enhanced_saliency_data:
        mapping = enhanced_saliency_data['prompt_saliency']
        for p in prompts:
            entry = mapping.get(p)
            prompt_saliency_dirs.append(entry.get('save_directory') if entry else None)
    else:
        prompt_saliency_dirs = [None] * len(prompts)

    initial_saliency_dir = None
    if isinstance(saliency_summary, dict):
        initial_saliency_dir = saliency_summary.get('save_directory')

    # Extract token attribution data for each enhanced prompt
    prompt_token_attributions = []
    if enhanced_saliency_data and 'prompt_saliency' in enhanced_saliency_data:
        mapping = enhanced_saliency_data['prompt_saliency']
        for p in prompts:
            entry = mapping.get(p)
            if entry and 'text_attribution' in entry:
                prompt_token_attributions.append(entry['text_attribution'])
            else:
                prompt_token_attributions.append(None)
    else:
        prompt_token_attributions = [None] * len(prompts)

    # Compute Final Query coordinates for each enhanced prompt (UMAP only)
    enhanced_final_query_coords = []
    for i, prompt in enumerate(prompts):
        # Recompute final query embedding for this enhanced prompt
        _, content_string = search_data['upload_contents'].split(',')
        decoded = base64.b64decode(content_string)
        tmp_coord = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp_coord.write(decoded); tmp_coord.close()
        
        try:
            from src.shared import cir_systems
            device_model = next(cir_systems.cir_system_searle.clip_model.parameters()).device
            img = Image.open(tmp_coord.name).convert('RGB')
            inp = cir_systems.cir_system_searle.preprocess(img).unsqueeze(0).to(device_model)
            
            with torch.no_grad():
                feat_raw = cir_systems.cir_system_searle.clip_model.encode_image(inp).float()
                feat = F.normalize(feat_raw, dim=-1)
                feat_np = feat.detach().cpu().numpy()
            
            # Compute Final Query coordinates only for UMAP
            xfq_enhanced = yfq_enhanced = None
            if cir_systems.cir_system_searle.eval_type in ['phi','searle','searle-xl']:
                pseudo = cir_systems.cir_system_searle.phi(feat_raw)
                cap = f"a photo of $ that {prompt}"
                tok = clip.tokenize([cap], context_length=77).to(device_model)
                from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
                final_feat = encode_with_pseudo_tokens(cir_systems.cir_system_searle.clip_model, tok, pseudo).float()
                final_feat = F.normalize(final_feat, dim=-1)
                final_query_feat_np = final_feat.detach().cpu().numpy().astype(np.float32)
                
                # Project to UMAP space
                umap_path = config.WORK_DIR / 'umap_reducer.pkl'
                pipeline_path = config.WORK_DIR / 'projection_pipeline.pkl'

                def _apply_projection_pipeline_enhanced(arr_np):
                    if not os.path.exists(pipeline_path):
                        return arr_np
                    try:
                        pipe = pickle.load(open(str(pipeline_path), 'rb'))
                        x = arr_np.copy()
                        if pipe.get('style_scaler') is not None:
                            x = pipe['style_scaler'].transform(x)
                            x = pipe['style_pca'].transform(x)
                            x = x[:, pipe['style_dims']]
                        if pipe.get('contrastive_scaler') is not None:
                            x_scaled = pipe['contrastive_scaler'].transform(x)
                            protos = pipe['contrastive_prototypes']
                            weight = pipe['contrastive_weight']
                            dists = ((protos - x_scaled)**2).sum(axis=1)
                            nearest_idx = int(dists.argmin())
                            proto_vec = protos[nearest_idx]
                            x_scaled = x_scaled + weight * (proto_vec - x_scaled)
                            x = x_scaled
                        if pipe.get('alt_scaler') is not None:
                            x = pipe['alt_scaler'].transform(x)
                            alt_model = pipe['alt_model']
                            x = alt_model.transform(x)
                        if pipe.get('final_pca') is not None:
                            x = pipe['final_pca'].transform(x)
                        return x
                    except Exception as e:
                        print(f"Warning: failed to apply projection pipeline: {e}")
                        return arr_np

                if os.path.exists(umap_path):
                    umap_reducer = pickle.load(open(str(umap_path), 'rb'))
                    proj_final = _apply_projection_pipeline_enhanced(final_query_feat_np)
                    final_umap_xy = umap_reducer.transform(proj_final)
                    xfq_enhanced, yfq_enhanced = float(final_umap_xy[0][0]), float(final_umap_xy[0][1])
            
            enhanced_final_query_coords.append({'x': xfq_enhanced, 'y': yfq_enhanced})
            os.unlink(tmp_coord.name)
            
        except Exception as e:
            print(f"Warning: failed to compute enhanced Final Query coords for prompt {i}: {e}")
            enhanced_final_query_coords.append({'x': None, 'y': None})
            try:
                os.unlink(tmp_coord.name)
            except:
                pass

    enhanced_prompts_data = {
        'prompts': prompts,
        'coverages': coverages,
        'mean_ranks': mean_ranks,
        'mean_sims': mean_sims,
        'ndcgs': ndcgs,
        'aps': aps,
        'mrrs': mrrs,
        'all_results': all_prompt_results,
        'best_idx': best_idx,
        'currently_viewing': best_idx,  # Default to showing best prompt results
        'prompt_saliency_dirs': prompt_saliency_dirs,
        'initial_saliency_dir': initial_saliency_dir,
        'prompt_token_attributions': prompt_token_attributions,  # Store in-memory token attribution data
        'enhanced_final_query_coords': enhanced_final_query_coords  # Store Final Query coordinates for each enhanced prompt
    }
    
    return status, enhance_children, enhanced_prompts_data

# New callback to handle enhanced prompt view button clicks
@callback(
    [Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data', allow_duplicate=True)],
    [Input({'type': 'enhanced-prompt-view', 'index': ALL}, 'n_clicks')],
    [State('cir-enhanced-prompts-data', 'data')],
    prevent_initial_call=True
)
def update_enhanced_prompt_view(n_clicks_list, enhanced_data):
    """Update the enhanced prompt results display when a view button is clicked"""
    if not enhanced_data or not any(n_clicks_list):
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the clicked button index
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        clicked_dict = json.loads(triggered_id)
        clicked_index = clicked_dict.get('index')
    except Exception:
        raise PreventUpdate
    
    # Update the currently viewing index
    enhanced_data['currently_viewing'] = clicked_index
    
    # Get data for the clicked prompt
    prompts = enhanced_data['prompts']
    coverages = enhanced_data['coverages']
    mean_ranks = enhanced_data['mean_ranks']
    mean_sims = enhanced_data['mean_sims']
    ndcgs = enhanced_data['ndcgs']
    aps = enhanced_data['aps']
    mrrs = enhanced_data['mrrs']
    all_results = enhanced_data['all_results']
    best_idx = enhanced_data['best_idx']
    
    clicked_prompt = prompts[clicked_index]
    clicked_coverage = coverages[clicked_index]
    clicked_mean_rank = mean_ranks[clicked_index]
    clicked_mean_sim = mean_sims[clicked_index]
    clicked_ndcg = ndcgs[clicked_index]
    clicked_ap = aps[clicked_index]
    clicked_mrr = mrrs[clicked_index]
    clicked_results = all_results[clicked_index]
    
    # Build image cards for the clicked prompt results
    df = Dataset.get()
    cards = []
    for img_name, score in clicked_results:
        img_path = None
        try:
            idx = int(img_name)
            if idx in df.index:
                img_path = df.loc[idx]['image_path']
        except:
            if img_name in df.index:
                img_path = df.loc[img_name]['image_path']
        if not img_path or not os.path.exists(img_path):
            continue
        data = open(img_path, "rb").read()
        src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        # Create card body with score (like original CIR)
        card_body = dbc.CardBody([
            html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
            html.P(f"Score: {score:.4f}", className='small text-center mt-1')
        ])
        card = dbc.Card(card_body, className='result-card', style={'cursor': 'default'})
        cards.append(dbc.Col(card, width=2))

    # Arrange cards into rows of up to 6
    rows = []
    for i in range(0, len(cards), 6):
        rows.append(dbc.Row(cards[i:i+6], className="g-2 mb-3"))
    images_grid = html.Div(rows)

    # Create enhanced table rows with updated button states
    table_rows = []
    for i, p in enumerate(prompts):
        cov = coverages[i]
        ndcg_val = ndcgs[i]
        ap_val = aps[i]
        # Highlight the currently viewing button
        if i == clicked_index:
            button_color = "primary"
            button_text = [html.I(className="fas fa-eye me-1"), "Viewing"]
        elif i == best_idx:
            button_color = "outline-success"
            button_text = [html.I(className="fas fa-crown me-1"), "Best"]
        else:
            button_color = "outline-primary"
            button_text = [html.I(className="fas fa-eye me-1"), "View"]
            
        view_button = dbc.Button(
            button_text,
            id={'type': 'enhanced-prompt-view', 'index': i},
            size="sm",
            color=button_color,
            className="btn-sm"
        )
        
        if i == best_idx:  # Highlight best prompt row
            row = html.Tr([
                html.Td([html.I(className="fas fa-crown text-warning me-2"), p], className="fw-bold"),
                html.Td([html.Span(f"{ndcg_val*100:.0f}%", className="badge bg-success")]),
                html.Td([html.Span(f"{ap_val*100:.0f}%", className="badge bg-success")]),
                html.Td([html.Span(f"{int(cov*100)}%", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(f"{ndcg_val*100:.0f}%", className="badge bg-secondary")),
                html.Td(html.Span(f"{ap_val*100:.0f}%", className="badge bg-secondary")),
                html.Td(html.Span(f"{int(cov*100)}%", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with updated buttons
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
            html.Th([html.I(className="fas fa-chart-line me-2"), "nDCG"], className="bg-light"),
            html.Th([html.I(className="fas fa-percentage me-2"), "AP"], className="bg-light"),
            html.Th([html.I(className="fas fa-bullseye me-2"), "Coverage"], className="bg-light"),
            html.Th([html.I(className="fas fa-cogs me-2"), "Actions"], className="bg-light")
        ]), className="thead-light"),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, responsive=True, className="mb-4 shadow-sm")

    # Build enhance results children with updated content
    enhance_children = [
        # Main title with icon
        html.Div([
            html.I(className="fas fa-brain text-primary me-2"),
            html.H5("Enhanced Prompt Analysis", className="d-inline text-primary")
        ], className="mt-3 mb-4"),
        
        # Candidates section
        html.Div([
            html.H6([html.I(className="fas fa-list-alt me-2"), "Candidate Prompts & Scores"], className="mb-3 text-secondary"),
            candidates_table
        ]),
        
        # Current viewing prompt highlight card
        dbc.Alert([
            html.H6([
                html.I(className="fas fa-trophy text-warning me-2") if clicked_index == best_idx else html.I(className="fas fa-eye text-info me-2"),
                "Currently Viewing" if clicked_index != best_idx else "Selected Best Prompt"
            ], className="alert-heading mb-2"),
            html.P(f'"{clicked_prompt}"', className="mb-1 font-monospace"),
            html.Small(f"nDCG: {clicked_ndcg*100:.0f}% | AP: {clicked_ap*100:.0f}% | Coverage: {clicked_coverage*100:.0f}%", className="text-muted")
        ], color="light" if clicked_index != best_idx else "light", 
           className="border-start border-info border-4" if clicked_index != best_idx else "border-start border-warning border-4"),
        
        # Results section
        html.Div([
            html.H6([html.I(className="fas fa-images me-2"), f"CIR Results for {'Best ' if clicked_index == best_idx else ''}Prompt"], className="mt-4 mb-3 text-secondary"),
            images_grid
        ])
    ]
    
    return enhance_children, enhanced_data

# Callback to populate the prompt enhancement tab with radio options
@callback(
    [Output('prompt-enhancement-content', 'children'),
     Output('prompt-selection', 'options'),
     Output('prompt-selection', 'value')],
    [Input('cir-enhanced-prompts-data', 'data'),
     Input('prompt-enh-fullscreen', 'data')],
    [State('prompt-selection', 'value')],
    prevent_initial_call=True
)
def populate_prompt_enhancement_tab(enhanced_data, is_fullscreen, current_selected_idx):
    """Populate the prompt enhancement tab when new enhanced prompts are available"""
    if not enhanced_data:
        # Show informational message when no enhancement data is available
        message_content = html.Div([
            html.I(className="fas fa-info-circle text-muted me-2"),
            html.Span("Run prompt enhancement to view enhanced prompts.", className="text-muted")
        ], className="d-flex align-items-center justify-content-center p-4")
        return message_content, [], None
    prompts = enhanced_data.get('prompts', [])
    coverages = enhanced_data.get('coverages', [])
    mean_ranks = enhanced_data.get('mean_ranks', [])
    mean_sims = enhanced_data.get('mean_sims', [])
    ndcgs = enhanced_data.get('ndcgs', [])
    aps = enhanced_data.get('aps', [])
    mrrs = enhanced_data.get('mrrs', [])
    best_idx = enhanced_data.get('best_idx')
    
    # Normalise fullscreen flag (None -> False)
    is_fullscreen = bool(is_fullscreen)
    
    # Create styled cards for each enhanced prompt
    cards = []
    for i, (prompt, cov, mean_rank, mean_sim, ndcg, ap, mrr) in enumerate(zip(prompts, coverages, mean_ranks, mean_sims, ndcgs, aps, mrrs)):
        is_best = (i == best_idx)
        is_selected = (i == current_selected_idx)
        
        # Card classes and styling - DO NOT apply 'selected' class directly
        # Let the CSS class management callbacks handle it dynamically
        card_classes = "prompt-enhancement-card"
        if is_best:
            card_classes += " best-prompt"
        # Remove this line that was causing the issue:
        # if is_selected:
        #     card_classes += " selected"
        
        icon_class = "fas fa-crown text-warning" if is_best else "fas fa-magic text-info"
        title_text = "Best" if is_best else f"#{i+1}"
        
        # Metric badges
        coverage_badge = html.Span(f"{cov*100:.0f}%", className="prompt-metric-badge bg-primary text-white")
        mean_rank_badge = html.Span(f"{mean_rank:.2f}", className="prompt-metric-badge bg-secondary text-white")
        mean_sim_badge = html.Span(f"{mean_sim:.4f}", className="prompt-metric-badge bg-success text-white")
        ndcg_badge = html.Span(f"{ndcg:.4f}", className="prompt-metric-badge bg-info text-white")
        ap_badge = html.Span(f"{ap:.4f}", className="prompt-metric-badge bg-warning text-white")
        mrr_badge = html.Span(f"{mrr:.4f}", className="prompt-metric-badge bg-dark text-white")
        
        # ----------------------------------------------------------------
        # Build metric badge row – show ALL metrics when fullscreen else
        # show the compact subset (Coverage, nDCG, AP).
        # ----------------------------------------------------------------
        metric_children = [
            html.Span("Coverage ", className="prompt-metric-label"),
            coverage_badge,
            html.Span(" nDCG ", className="prompt-metric-label", style={'marginLeft': '0.3rem'}),
            ndcg_badge,
            html.Span(" AP ", className="prompt-metric-label", style={'marginLeft': '0.3rem'}),
            ap_badge,
        ]

        if is_fullscreen:
            # Append the extra metrics when in fullscreen
            metric_children.extend([
                html.Span(" Mean Rank ", className="prompt-metric-label", style={'marginLeft': '0.3rem'}),
                mean_rank_badge,
                html.Span(" Mean Sim ", className="prompt-metric-label", style={'marginLeft': '0.3rem'}),
                mean_sim_badge,
                html.Span(" MRR ", className="prompt-metric-label", style={'marginLeft': '0.3rem'}),
                mrr_badge,
            ])
        
        card = html.Div([
            dbc.Card([
                dbc.CardBody([
                    # Header with title and metrics on same line
                    html.Div([
                        html.I(className=f"{icon_class} prompt-card-icon"),
                        html.Span(title_text, className="prompt-card-title", style={'marginRight': '0.5rem'}),
                        html.Div(metric_children, className="prompt-card-metrics", style={'display': 'inline-flex', 'alignItems': 'center'})
                    ], className="prompt-card-header", style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap'}),
                    
                    # Prompt text with improved styling
                    html.P(
                        f'"{prompt}"', 
                        className=f"prompt-card-text{' best-prompt' if is_best else ''}"
                    )
                ], className="prompt-card-body")
            ], className=card_classes, style={'cursor': 'pointer'})
        ], id={'type': 'prompt-card', 'index': i}, n_clicks=0)
        cards.append(card)
    
    # Combine radio items for callback wiring (enhanced prompts only)
    all_options = [{'label': f"Enhanced Prompt {i+1}", 'value': i} for i in range(len(prompts))]
    
    content = [
        html.Div([
            html.I(className="fas fa-list-alt text-primary me-2", style={'fontSize': '0.9rem'}),
            html.H6("Enhanced Prompts", className="text-primary mb-2", style={'fontSize': '0.85rem', 'display': 'inline'})
        ], style={'marginBottom': '0.5rem'}),
        html.Div(cards, className="prompt-cards-container")
    ]
    
    return content, all_options, current_selected_idx

# Callback to handle prompt card clicks and update selection
@callback(
    [Output('prompt-selection', 'value', allow_duplicate=True),
     Output({'type': 'prompt-card', 'index': ALL}, 'className')], # Removed style output
    Input({'type': 'prompt-card', 'index': ALL}, 'n_clicks'),
    [State('prompt-selection', 'value'),
     State('cir-enhanced-prompts-data', 'data'),
     State('viz-mode', 'data'),
     State({'type': 'prompt-card', 'index': ALL}, 'className')],
    prevent_initial_call=True
)
def handle_prompt_card_selection(n_clicks_list, current_value, enhanced_data, viz_mode, current_classnames):
    """Handle card clicks for prompt selection"""
    if not any(n_clicks_list) or not enhanced_data:
        raise PreventUpdate
    
    # If visualization mode is ON, ignore prompt-enhancement selection logic
    if viz_mode:
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the clicked card index
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        clicked_dict = json.loads(triggered_id)
        clicked_index = clicked_dict.get('index')
    except Exception:
        raise PreventUpdate
    
    # Deselect if clicking the same prompt twice
    if clicked_index == current_value:
        new_selected = -1
    else:
        new_selected = clicked_index
    
    prompts = enhanced_data.get('prompts', []) # Need prompts to iterate correctly
    best_idx = enhanced_data.get('best_idx') # Need best_idx for class names
    new_classnames = [] # This will hold the new class strings

    # Loop through all prompt cards to determine their new class names
    for i in range(len(prompts)): # Iterate through all possible indices
        is_best = (i == best_idx)
        is_selected = (i == new_selected)
        
        # --- Determine new class names ---
        class_parts = ["prompt-enhancement-card"] # Start with base class
        if is_best:
            class_parts.append("best-prompt")
        if is_selected:
            class_parts.append("selected")
        new_classnames.append(" ".join(class_parts))

    return new_selected, new_classnames

# Callback to update all widgets when an enhanced prompt is selected
@callback(
    [Output('gallery', 'children', allow_duplicate=True),
     Output('wordcloud', 'list', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('prompt-selection', 'value'),
    [State('cir-enhanced-prompts-data', 'data'),
     State('cir-search-data', 'data'),
     State('cir-toggle-state', 'data'),
     State('scatterplot', 'figure'),
     State('selected-gallery-image-ids', 'data'),
     State('viz-mode', 'data')],
    prevent_initial_call=True
)
def update_widgets_for_enhanced_prompt(selected_idx, enhanced_data, search_data, cir_toggle_state, scatterplot_fig, selected_gallery_image_ids, viz_mode):
    """Update widgets for original or enhanced prompt selection, recomputing final query for enhanced and allowing revert."""
    if not cir_toggle_state or selected_idx is None or (enhanced_data is None and selected_idx != -1):
        raise PreventUpdate
    df = Dataset.get()
    # Determine axis and base query coords
    axis_title = scatterplot_fig['layout']['xaxis']['title']['text']
    xq = search_data.get('umap_x_query'); yq = search_data.get('umap_y_query')
    xfq = yfq = None
    if axis_title != 'umap_x':
        xq, yq = None, None
    # Handle original revert
    if selected_idx == -1:
        topk_ids = search_data.get('topk_ids', [])
        top1_id = search_data.get('top1_id')
        xfq = search_data.get('umap_x_final_query'); yfq = search_data.get('umap_y_final_query')
        if axis_title != 'umap_x':
            xfq, yfq = None, None
    else:
        # Enhanced prompt branch
        prompts = enhanced_data.get('prompts', [])
        results_list = enhanced_data.get('all_results', [])
        if selected_idx >= len(results_list):
            raise PreventUpdate
        selected_results = results_list[selected_idx]
        # Build top-k IDs
        topk_ids = []
        for img_name, _ in selected_results:
            try: idx = int(img_name)
            except: idx = img_name
            if idx in df.index:
                topk_ids.append(idx)
        top1_id = topk_ids[0] if topk_ids else None
        # Use pre-computed Final Query coordinates from enhanced prompts data
        # No need to recompute - we already calculated these during prompt enhancement!
        xfq = yfq = None
        if axis_title == 'umap_x' and enhanced_data:
            enhanced_coords = enhanced_data.get('enhanced_final_query_coords', [])
            if selected_idx < len(enhanced_coords):
                coord_data = enhanced_coords[selected_idx]
                xfq, yfq = coord_data.get('x'), coord_data.get('y')
    # Scatterplot updates now handled by unified controller
    # Wordcloud
    counts = df.loc[topk_ids]['class_name'].value_counts()
    if len(counts):
        wg = wordcloud.wordcloud_weight_rescale(counts.values,1,counts.max())
        wc = sorted([[cn,w] for cn,w in zip(counts.index,wg)],key=lambda x:x[1],reverse=True)
    else: wc=[]
    # Gallery and Histogram
    cir_df = df.loc[topk_ids]
    # Clear any previous gallery selections when switching to enhanced prompt results
    gal = gallery.create_gallery_children(cir_df['image_path'].values,cir_df['class_name'].values,cir_df.index.values,[])
    hist = histogram.draw_histogram(cir_df)
    return gal, wc, hist, None, []



# -----------------------------------------------------------------------------
# Helper – Re-use card–building logic for Query Results so it can be invoked by
#           multiple callbacks (baseline query and enhanced prompt views).
# -----------------------------------------------------------------------------

# Helper now also receives viz_mode so that the Visualize toggle button is rendered
# with the correct ON/OFF label and color each time the Results layout is rebuilt.
def _build_query_results_layout(result_tuples, *, clickable: bool = True, viz_mode: bool = False, preselected_ids=None):
    """Return a Dash HTML div containing the grid of result cards.

    Parameters
    ----------
    result_tuples : list[(str,float)]
        List of (image_id, similarity_score) pairs as produced by the CIR system.
    """

    df = Dataset.get()
    from src.callbacks.saliency_callbacks import load_and_resize_image  # local import

    # Normalise *preselected_ids* to a set of strings for fast lookup
    preselected_set = set(str(x) for x in (preselected_ids or []))

    cards = []
    for idx, (img_name, score) in enumerate(result_tuples):
        img_path = None
        # Try numeric index first, then string
        try:
            int_idx = int(img_name)
            if int_idx in df.index:
                img_path = df.loc[int_idx]["image_path"]
        except Exception:
            if img_name in df.index:
                img_path = df.loc[img_name]["image_path"]

        if not img_path or not os.path.exists(img_path):
            continue

        # thumbnail (150×150 max)
        try:
            src = load_and_resize_image(img_path, max_width=150, max_height=150)
            if not src:
                continue
        except Exception:
            continue

        card_body = dbc.CardBody([
            html.Img(
                src=src,
                className="img-fluid",
                style={
                    "maxWidth": "100%",
                    "height": "auto",
                    "objectFit": "contain",
                    "borderRadius": "4px",
                    "marginBottom": "8px",
                },
            ),
            html.Div([
                html.Small(
                    f"{score:.3f}",
                    className="badge bg-primary",
                    style={"fontSize": "0.65rem", "fontWeight": "500"},
                )
            ], className="text-center"),
        ], style={"padding": "8px", "display": "flex", "flexDirection": "column", "alignItems": "center", "justifyContent": "center"})

        # TOP-1 card behaviour differs between modes
        if idx == 0:
            inner = dbc.Card(card_body, className="result-card", style={"border": "1px solid #dee2e6", "borderRadius": "6px", "height": "100%"})

            if viz_mode:
                # Clickable wrapper (multi-select in visualization)
                wrapper = html.Div([
                    inner,
                    html.Div("TOP-1", className="badge bg-warning position-absolute top-0 start-0 m-1"),
                ], id={"type": "cir-result-card", "index": str(img_name)}, n_clicks=0,
                className="result-card-wrapper position-relative", style={"height": "100%"})
            else:
                # Non-clickable in prompt-enhancement mode
                wrapper = html.Div([
                    inner,
                    html.Div("TOP-1", className="badge bg-warning position-absolute top-0 start-0 m-1"),
                ], className="result-card-wrapper position-relative", style={"cursor": "default", "height": "100%"})
        else:
            inner = dbc.Card(card_body, className="result-card", style={"border": "1px solid #dee2e6", "borderRadius": "6px", "height": "100%"})
            # Highlight if this image was previously selected
            preselected = str(img_name) in preselected_set

            if clickable:
                # Interactive wrapper (used in prompt-enhancement mode)
                wrapper = html.Div(
                    inner,
                    id={"type": "cir-result-card", "index": str(img_name)},
                    n_clicks=0,
                    className=f"result-card-wrapper{' selected' if preselected else ''}",
                    style={"height": "100%"},
                )
            else:
                # Non-interactive wrapper for enhanced prompt results
                wrapper = html.Div(
                    inner,
                    className=f"result-card-wrapper{' selected' if preselected else ''}",
                    style={"height": "100%"},
                )

        cards.append(wrapper)

    rows = []
    for i in range(0, len(cards), 4):
        chunk = cards[i : i + 4]
        cols = [dbc.Col(c, width=3, className="mb-3 px-2") for c in chunk]
        rows.append(dbc.Row(cols, className="g-2"))

    # Visualize toggle reflects current viz_mode
    btn_children = "Visualize ON" if viz_mode else "Visualize OFF"
    btn_color = "success" if viz_mode else "secondary"

    header = html.Div([
        html.H5("Retrieved Images", className="mb-0", style={"display": "inline-block"}),
        dbc.Button(id="visualize-toggle-button", size="sm", color=btn_color, class_name="ms-2", n_clicks=0,
                   children=btn_children)
    ], className="d-flex align-items-center mb-3")

    return html.Div([header] + rows)


# -----------------------------------------------------------------------------
# Callback – switch Query Results when an enhanced prompt is selected / deselected
# -----------------------------------------------------------------------------


@callback(
    [Output("cir-results", "children", allow_duplicate=True),
     Output("enhance-prompt-button", "disabled", allow_duplicate=True),
     Output("enhance-prompt-button", "color", allow_duplicate=True)],
    Input("prompt-selection", "value"),
    [State("cir-enhanced-prompts-data", "data"), State("cir-search-data", "data"), State("viz-mode", "data"),
     State('cir-selected-image-ids', 'data')],
    prevent_initial_call=True,
)
def update_query_results_for_prompt_selection(selected_idx, enhanced_data, search_data, viz_mode, selected_ids):
    """Render top-k images of the selected enhanced prompt (or original query).

    • When *selected_idx* >= 0 we display the corresponding enhanced-prompt results
      and disable the "Enhance prompt" button.
    • When *selected_idx* == -1 we revert to the baseline query results and leave
      the button state to other callbacks (so it can become enabled again once a
      result card is picked).
    """

    if selected_idx is None or search_data is None:
        raise PreventUpdate

    # If enhanced prompt selected and we have data
    if selected_idx >= 0 and enhanced_data is not None:
        results_lists = enhanced_data.get("all_results", [])
        if selected_idx >= len(results_lists):
            raise PreventUpdate
        results = results_lists[selected_idx]
        # Non-TOP-1 images become clickable when visualization mode is ON so that
        # they can be multi-selected on the scatterplot.
        layout = _build_query_results_layout(results, clickable=viz_mode, viz_mode=viz_mode)
        return layout, True, "secondary"

    # Revert to original baseline query results when selected_idx == -1
    if selected_idx == -1:
        original_results = search_data.get("original_results")
        if not original_results:
            raise PreventUpdate
        # Highlight previously selected ideal images when returning to baseline results
        pre_ids = selected_ids if not viz_mode else None
        layout = _build_query_results_layout([(name, score) for name, score in original_results], clickable=True, viz_mode=viz_mode, preselected_ids=pre_ids)
        # Do **not** change enhance-prompt button state here – let other callbacks
        # (image-selection) manage it. Hence we use no_update.
        from dash import no_update
        return layout, no_update, no_update

    raise PreventUpdate

# -----------------------------------------------------------------------------
# Visualization mode toggle
# -----------------------------------------------------------------------------

@callback(
    [Output('viz-mode', 'data'),
     Output('visualize-toggle-button', 'children'),
     Output('visualize-toggle-button', 'color'),
     Output('cir-results', 'children', allow_duplicate=True),
     Output('cir-selected-image-ids', 'data', allow_duplicate=True),
     Output('viz-selected-ids', 'data', allow_duplicate=True),
],
    Input('visualize-toggle-button', 'n_clicks'),
    [State('viz-mode', 'data'),
     State('prompt-selection', 'value'),
     State('cir-enhanced-prompts-data', 'data'),
     State('cir-search-data', 'data'),
     State('scatterplot', 'figure')],
    prevent_initial_call=True
)
def toggle_visualize_mode(n_clicks, current_mode, selected_idx, enhanced_data, search_data, scatterplot_fig):
    """Toggle visualization mode ON/OFF and clear all selections when switching modes."""
    # Only toggle if we actually have a click (n_clicks > 0)
    if n_clicks is None or n_clicks == 0:
        # Initial state - keep current mode and set appropriate label
        label = 'Visualize ON' if current_mode else 'Visualize OFF'
        color = 'success' if current_mode else 'secondary'
        from dash import no_update
        return current_mode, label, color, no_update, no_update, no_update
    
    # Toggle the mode
    new_mode = not current_mode
    label = 'Visualize ON' if new_mode else 'Visualize OFF'
    color = 'success' if new_mode else 'secondary'

    # Clear all selections when switching modes
    cleared_cir_selected = []
    cleared_viz_selected = []

    # Scatterplot updates now handled by unified controller

    # Rebuild Query Results layout with new viz_mode state
    if search_data is None:
        return new_mode, label, color, no_update, cleared_cir_selected, cleared_viz_selected

    # Determine which results to show (baseline or selected enhanced prompt)
    if selected_idx is not None and selected_idx >= 0 and enhanced_data is not None:
        rs_lists = enhanced_data.get('all_results', [])
        if selected_idx < len(rs_lists):
            res = rs_lists[selected_idx]
            # Enable clicking of non-TOP-1 cards only when visualization mode is ON
            layout = _build_query_results_layout(res, clickable=new_mode, viz_mode=new_mode)
        else:
            layout = no_update
    else:
        orig = search_data.get('original_results')
        if orig:
            layout = _build_query_results_layout([(n, s) for n, s in orig], clickable=True, viz_mode=new_mode)
        else:
            layout = no_update

    return new_mode, label, color, layout, cleared_cir_selected, cleared_viz_selected

# -----------------------------------------------------------------------------
# React to viz-mode changes – clear selections when turning OFF
# -----------------------------------------------------------------------------

@callback(
    [Output({'type': 'cir-result-card', 'index': ALL}, 'className', allow_duplicate=True),
     Output('viz-selected-ids', 'data', allow_duplicate=True)],
    Input('viz-mode', 'data'),
    [State({'type': 'cir-result-card', 'index': ALL}, 'className'),
     State('viz-selected-ids', 'data'),
     State('scatterplot', 'figure')],
    prevent_initial_call=True
)
def handle_viz_mode_change(viz_mode, current_classnames, selected_ids, scatterplot_fig):
    """When visualization mode is toggled OFF, clear selected ids and highlights."""
    if viz_mode:
        # Turning ON – keep current selections/highlights
        raise PreventUpdate

    # Turning OFF – remove visual-selected class (scatterplot handled by unified controller)
    new_classnames = []
    for cls in current_classnames:
        parts = cls.split()
        if 'visual-selected' in parts:
            parts.remove('visual-selected')
        new_classnames.append(' '.join(parts))

    return new_classnames, []

# -----------------------------------------------------------------------------
# Selection of images while Visualization mode is ON (multi-select capability)
# -----------------------------------------------------------------------------

@callback(
    [Output({'type': 'cir-result-card', 'index': ALL}, 'className', allow_duplicate=True),
     Output('viz-selected-ids', 'data', allow_duplicate=True)],
    Input({'type': 'cir-result-card', 'index': ALL}, 'n_clicks'),
    [State({'type': 'cir-result-card', 'index': ALL}, 'className'),
     State('viz-mode', 'data'),
     State('viz-selected-ids', 'data'),
     State('scatterplot', 'figure')],
    prevent_initial_call=True
)
def select_images_for_visualization(n_clicks_list, current_classnames, viz_mode, selected_ids, scatterplot_fig):
    """Handle multi-selection of result cards while visualization mode is ON."""
    # Only act when visualization mode is active
    if not viz_mode:
        raise PreventUpdate

    # If no card was actually clicked do nothing
    if not any(n_clicks_list):
        raise PreventUpdate

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    # Ignore automatic callbacks during initial layout build (n_clicks == 0)
    triggered_id_raw = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        trig_dict = json.loads(triggered_id_raw)
        clicked_id = str(trig_dict.get('index'))
    except Exception:
        raise PreventUpdate

    # Determine n_clicks value for the triggered card
    clicked_n = None
    for inp_dict, n_val in zip(ctx.inputs_list[0], n_clicks_list):
        if str(inp_dict['id']['index']) == clicked_id:
            clicked_n = n_val
            break

    if clicked_n is None or clicked_n == 0:
        # Spurious callback during layout creation – ignore
        raise PreventUpdate

    # Initialize list
    selected_ids = selected_ids or []

    # Toggle selection
    if clicked_id in selected_ids:
        selected_ids.remove(clicked_id)
    else:
        selected_ids.append(clicked_id)

    # ---------------------------------------------------------------------
    # Update className list – add/remove visual-selected class
    # ---------------------------------------------------------------------
    new_classnames = []
    selected_set = set(selected_ids)
    # Iterate over inputs to preserve order and access IDs directly
    for input_dict, cls in zip(ctx.inputs_list[0], current_classnames):
        wid = str(input_dict['id']['index'])
        parts = cls.split()
        if wid in selected_set and 'visual-selected' not in parts:
            parts.append('visual-selected')
        if wid not in selected_set and 'visual-selected' in parts:
            parts.remove('visual-selected')
        new_classnames.append(' '.join(parts))

    # Scatterplot updates now handled by unified controller
    return new_classnames, selected_ids

# -----------------------------------------------------------------------------
# Clear selections when the global "Visualize CIR results" / "Hide CIR results"
# button is clicked while Visualization mode is currently ON.
# -----------------------------------------------------------------------------

@callback(
    [Output({'type': 'cir-result-card', 'index': ALL}, 'className', allow_duplicate=True),
     Output('viz-selected-ids', 'data', allow_duplicate=True)],
    Input('cir-toggle-button', 'n_clicks'),
    [State('viz-mode', 'data'),
     State({'type': 'cir-result-card', 'index': ALL}, 'className'),
     State('viz-selected-ids', 'data'),
     State('scatterplot', 'figure')],
    prevent_initial_call=True,
)
def clear_visual_selections_on_cir_toggle(n_clicks, viz_mode, current_classnames, selected_ids, scatterplot_fig):
    """If visualization mode is ON and there are selected images, deselect them when
    the user clicks the global CIR results visibility button (either direction).
    This ensures that any previous highlight is cleared regardless of whether the
    button is switching *into* or *out of* the visualization overlay.
    """
    from dash import no_update
    import copy

    # Coerce potential None values for robustness
    current_classnames = current_classnames or []
    selected_ids = selected_ids or []

    # Only act when viz-mode is ON and there are selections.
    if not viz_mode or not selected_ids:
        # Nothing to clear → keep current state.
        raise PreventUpdate

    # ------------------------------------------------------------------
    # Remove the "visual-selected" CSS class from all result-card wrappers
    # ------------------------------------------------------------------
    new_classnames = []
    for cls in current_classnames:
        parts = cls.split()
        if 'visual-selected' in parts:
            parts.remove('visual-selected')
        new_classnames.append(' '.join(parts))

    # Scatterplot updates now handled by unified controller
    return new_classnames, []

# -----------------------------------------------------------------------------
# Loading visualisation for Prompt Enhancement
# -----------------------------------------------------------------------------

@callback(
    Output('prompt-enhancement-content', 'children', allow_duplicate=True),
    Input('enhance-prompt-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_prompt_enhancement_loading(n_clicks):
    """Display a spinner + friendly message while prompt enhancement is running.

    This callback fires immediately when the user clicks the *Enhance prompt*
    button and renders a loading visual inside the **Prompt Enhancement** card.
    Once the heavy `enhance_prompt` callback finishes, the regular
    `populate_prompt_enhancement_tab` callback will overwrite this content with
    the actual results, so we mark this output as *allow_duplicate=True*.
    """
    from dash.exceptions import PreventUpdate
    if not n_clicks:
        raise PreventUpdate

    return html.Div([
        dbc.Spinner(color="primary", type="grow", size="lg", spinnerClassName="mb-3"),
        html.Span("Generating enhanced prompts…", className="text-muted fw-semibold")
    ], className="d-flex flex-column align-items-center justify-content-center p-4")

# -----------------------------------------------------------------------------
# Loading visualisation for normal CIR search (Query Results component)
# -----------------------------------------------------------------------------

@callback(
    Output('cir-results', 'children', allow_duplicate=True),
    Input('cir-search-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_cir_search_loading(n_clicks):
    """Display a spinner & message in the Query Results card while the CIR
    search is running. It is intentionally lightweight and will be overwritten
    by `perform_cir_search` once the results are ready."""
    from dash.exceptions import PreventUpdate
    if not n_clicks:
        raise PreventUpdate

    return html.Div([
        dbc.Spinner(color="primary", type="border", size="lg", spinnerClassName="mb-3"),
        html.Span("Retrieving images…", className="text-muted fw-semibold")
    ], className="d-flex flex-column align-items-center justify-content-center p-4")

# Callback to apply selected class after UI rebuild (e.g., during fullscreen toggle)
@callback(
    Output({'type': 'prompt-card', 'index': ALL}, 'className', allow_duplicate=True),
    [Input('prompt-enh-fullscreen', 'data'),
     Input('cir-enhanced-prompts-data', 'data')],
    [State('prompt-selection', 'value'),
     State({'type': 'prompt-card', 'index': ALL}, 'className')],
    prevent_initial_call=True
)
def apply_selected_class_after_rebuild(is_fullscreen, enhanced_data, current_selected_idx, current_classnames):
    """Apply the selected class to the appropriate prompt card after UI rebuild (e.g., fullscreen toggle)"""
    if not enhanced_data or current_selected_idx is None:
        raise PreventUpdate
    
    prompts = enhanced_data.get('prompts', [])
    best_idx = enhanced_data.get('best_idx')
    
    # Build the correct class names for all cards
    new_classnames = []
    for i in range(len(prompts)):
        is_best = (i == best_idx)
        is_selected = (i == current_selected_idx and current_selected_idx >= 0)
        
        class_parts = ["prompt-enhancement-card"]
        if is_best:
            class_parts.append("best-prompt")
        if is_selected:
            class_parts.append("selected")
        new_classnames.append(" ".join(class_parts))
    
    return new_classnames

# Callback to clear prompt card selection when deselect button is pressed
@callback(
    Output({'type': 'prompt-card', 'index': ALL}, 'className', allow_duplicate=True),
    Input('prompt-selection', 'value'),
    [State('cir-enhanced-prompts-data', 'data'),
     State({'type': 'prompt-card', 'index': ALL}, 'className')],
    prevent_initial_call=True
)
def clear_prompt_selection_on_deselect(selected_idx, enhanced_data, current_classnames):
    """Clear prompt card selection when prompt-selection becomes -1 (deselected)"""
    if not enhanced_data or selected_idx != -1:
        raise PreventUpdate
    
    prompts = enhanced_data.get('prompts', [])
    best_idx = enhanced_data.get('best_idx')
    
    # Build class names without any selection
    new_classnames = []
    for i in range(len(prompts)):
        is_best = (i == best_idx)
        
        class_parts = ["prompt-enhancement-card"]
        if is_best:
            class_parts.append("best-prompt")
        # No "selected" class since we're deselecting
        new_classnames.append(" ".join(class_parts))
    
    return new_classnames

def extract_class_and_style_info(selected_image_ids: List[str]) -> Dict[str, List[str]]:
    """
    Extract class and style information from selected image IDs.
    
    Args:
        selected_image_ids: List of image IDs in string format
        
    Returns:
        Dictionary with 'classes' and 'styles' lists
    """
    df = Dataset.get()
    classes = []
    styles = []
    
    for image_id in selected_image_ids:
        try:
            # Convert to integer if it's a string representation of an integer
            if isinstance(image_id, str) and image_id.isdigit():
                image_id = int(image_id)
            
            # Get the image information from the dataset
            if image_id in df.index:
                row = df.loc[image_id]
                
                # Extract class name
                class_name = row['class_name']
                classes.append(class_name)
                
                # Extract style from filename (format: style_number.jpg)
                image_path = row['image_path']
                filename = os.path.basename(image_path)
                # Remove file extension and extract style (first part before underscore)
                base_name = os.path.splitext(filename)[0]
                if '_' in base_name:
                    style = base_name.split('_')[0]
                    styles.append(style)
                else:
                    # If no underscore, use the whole base name as style
                    styles.append(base_name)
                    
        except Exception as e:
            print(f"Warning: Could not extract info for image ID {image_id}: {e}")
            continue
    
    # Remove duplicates while preserving order
    unique_classes = list(dict.fromkeys(classes))
    unique_styles = list(dict.fromkeys(styles))
    
    return {
        'classes': unique_classes,
        'styles': unique_styles
    }