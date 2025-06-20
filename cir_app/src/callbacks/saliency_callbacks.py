import os
import base64
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import glob
from PIL import Image
import io

from dash import callback, Input, Output, State, html, dcc, callback_context
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

from src import config

# Global cache for processed saliency images
_saliency_image_cache = {}
_current_pairs = []

# Add import for Plotly
import plotly.graph_objects as go

def clear_saliency_cache():
    """Clear the saliency image cache to free memory."""
    global _saliency_image_cache
    _saliency_image_cache.clear()

def find_saliency_pairs(save_directory: str) -> List[Tuple[int, str, str, str]]:
    """
    Find saliency image pairs in the save directory.
    
    Args:
        save_directory: Path to the saliency output directory
        
    Returns:
        List of tuples (rank, candidate_path, reference_path, image_name)
        Sorted by rank
    """
    if not save_directory or not os.path.exists(save_directory):
        return []
    
    save_path = Path(save_directory)
    pairs = []
    
    # Look for candidate saliency files (result_*_heatmap_*.png, but NOT ref_heatmap)
    pattern = "result_*_heatmap_*.png"
    candidate_files = [f for f in save_path.glob(pattern) if "_ref_heatmap_" not in f.name]
    
    for candidate_file in candidate_files:
        filename = candidate_file.name
        # Parse filename: result_{rank}_heatmap_{image_name}.png
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 3 and parts[0] == 'result':
            try:
                rank = int(parts[1])
                # Extract the image name from the candidate file
                # Format: result_{rank}_heatmap_{image_name}.png
                heatmap_index = -1
                for i, part in enumerate(parts):
                    if part == 'heatmap':
                        heatmap_index = i
                        break
                
                if heatmap_index >= 0 and heatmap_index + 1 < len(parts):
                    image_name_parts = parts[heatmap_index + 1:]
                    image_name = '_'.join(image_name_parts)
                    
                    # Construct the corresponding reference file name
                    ref_filename = f"result_{rank}_ref_heatmap_{image_name}.png"
                    ref_file = save_path / ref_filename
                    
                    if ref_file.exists():
                        pairs.append((rank, str(candidate_file), str(ref_file), image_name))
                        
            except (ValueError, IndexError):
                continue
    
    # Sort by rank
    pairs.sort(key=lambda x: x[0])
    return pairs


def load_and_resize_image(image_path: str, max_width: int = 500, max_height: int = 350) -> Optional[str]:
    """
    Load image, resize it for display, and convert to base64 data URL.
    Uses caching to avoid re-processing the same image.
    
    Args:
        image_path: Path to the image file
        max_width: Maximum width for display
        max_height: Maximum height for display
    
    Returns:
        Base64 data URL or None if error
    """
    global _saliency_image_cache
    
    # Create cache key
    cache_key = f"{image_path}_{max_width}_{max_height}"
    
    # Check cache first
    if cache_key in _saliency_image_cache:
        return _saliency_image_cache[cache_key]
    
    try:
        if not os.path.exists(image_path):
            return None
            
        # Load and resize image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate resize dimensions while maintaining aspect ratio
            img_width, img_height = img.size
            aspect_ratio = img_width / img_height
            
            if img_width > max_width or img_height > max_height:
                if aspect_ratio > 1:  # Wider than tall
                    new_width = min(max_width, img_width)
                    new_height = int(new_width / aspect_ratio)
                else:  # Taller than wide
                    new_height = min(max_height, img_height)
                    new_width = int(new_height * aspect_ratio)
                
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to base64 with JPEG compression
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=80, optimize=True)
            buffer.seek(0)
            
            data = buffer.getvalue()
            base64_str = base64.b64encode(data).decode()
            result = f"data:image/jpeg;base64,{base64_str}"
            
            # Cache the result (but limit cache size)
            if len(_saliency_image_cache) > 15:  # Limit cache size
                # Remove oldest entries
                keys_to_remove = list(_saliency_image_cache.keys())[:5]
                for key in keys_to_remove:
                    del _saliency_image_cache[key]
            
            _saliency_image_cache[cache_key] = result
            return result
            
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


@callback(
    Output('saliency-data', 'data', allow_duplicate=True),
    Input('saliency-data', 'data'),
    prevent_initial_call=True
)
def preprocess_saliency_data(saliency_data):
    """
    Preprocess saliency data by finding pairs and clearing cache.
    This runs once when new saliency data is loaded.
    """
    global _current_pairs
    
    # Clear cache when new data arrives
    clear_saliency_cache()
    
    if not saliency_data or not saliency_data.get('save_directory'):
        _current_pairs = []
        return saliency_data
    
    # Find pairs once and store globally
    _current_pairs = find_saliency_pairs(saliency_data['save_directory'])
    
    # Add pairs count to saliency data for easier access
    if saliency_data:
        saliency_data['pairs_count'] = len(_current_pairs)
    
    return saliency_data


@callback(
    [Output('saliency-content', 'children'),
     Output('saliency-navigation', 'style'),
     Output('saliency-current-info', 'children'),
     Output('saliency-prev-btn', 'disabled'),
     Output('saliency-next-btn', 'disabled')],
    [Input('saliency-data', 'data'),
     Input('saliency-current-index', 'data'),
     Input('cir-toggle-state', 'data'),
     Input('saliency-fullscreen', 'data')],
    prevent_initial_call=True
)
def update_saliency_display(saliency_data, current_index, cir_toggle_state, saliency_fullscreen):
    """Update the saliency display based on current data and index."""
    global _current_pairs
    
    # Only show saliency when CIR results are being visualized
    if not cir_toggle_state:
        return ([
            html.Div([
                html.I(className="fas fa-eye-slash text-muted me-2"),
                html.Span("Enable visualization to view saliency", className="text-muted small")
            ], className="text-center p-2")
        ], {'display': 'none'}, "", True, True)
    
    if not saliency_data or not saliency_data.get('save_directory'):
        return ([
            html.Div([
                html.I(className="fas fa-brain text-muted me-2"),
                html.Span("No saliency data available", className="text-muted small")
            ], className="text-center p-2")
        ], {'display': 'none'}, "", True, True)
    
    # Use cached pairs
    pairs = _current_pairs
    
    if not pairs:
        return ([
            html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.Span("No saliency maps found", className="text-muted small")
            ], className="text-center p-2")
        ], {'display': 'none'}, "", True, True)
    
    # Ensure current_index is within bounds
    if current_index < 0:
        current_index = 0
    elif current_index >= len(pairs):
        current_index = len(pairs) - 1
    
    # Get current pair
    rank, candidate_path, reference_path, image_name = pairs[current_index]
    
    # Use larger thumbnails when the saliency panel is in fullscreen
    if saliency_fullscreen:
        mw = 350; mh = 350
    else:
        mw = 150; mh = 150

    # Adjust container size based on fullscreen
    container_style = {
        'minHeight': '360px',  # taller box for bigger image
        'maxHeight': '480px',
        'width': '100%',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center'
    } if saliency_fullscreen else {}
    
    reference_src = load_and_resize_image(reference_path, max_width=mw, max_height=mh)
    candidate_src = load_and_resize_image(candidate_path, max_width=mw, max_height=mh)
    
    if not reference_src or not candidate_src:
        # If images fail to load, show compact error
        content = [
            html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.Span("Error loading images", className="text-muted small")
            ], className="text-center p-2")
        ]
    else:
        # Create compact display content
        content = [
            html.Div([
                # Height-centered images (no rank badge here anymore)
                html.Div([
                    # Reference image
                    html.Div([
                        html.Div("Reference", className="text-center text-primary fw-bold mb-1", style={'fontSize': '0.65rem'}),
                        html.Div([
                            html.Img(
                                src=reference_src,
                                className="saliency-compact-image"
                            )
                        ], className="saliency-compact-container reference-saliency", style=container_style)
                    ], className="saliency-column"),
                    
                    # Candidate image  
                    html.Div([
                        html.Div("Candidate", className="text-center text-success fw-bold mb-1", style={'fontSize': '0.65rem'}),
                        html.Div([
                            html.Img(
                                src=candidate_src,
                                className="saliency-compact-image"
                            )
                        ], className="saliency-compact-container candidate-saliency", style=container_style)
                    ], className="saliency-column")
                ], className="saliency-pair-row")
            ], className="saliency-compact-content")
        ]
        
    # Navigation info - only show rank
    nav_info = f"Rank {rank}"
    
    # Navigation buttons state
    prev_disabled = current_index <= 0
    next_disabled = current_index >= len(pairs) - 1
    
    return (content, 
            {'display': 'block'}, 
            nav_info, 
            prev_disabled, 
            next_disabled)


@callback(
    Output('saliency-current-index', 'data'),
    [Input('saliency-prev-btn', 'n_clicks'),
     Input('saliency-next-btn', 'n_clicks')],
    [State('saliency-current-index', 'data'),
     State('saliency-data', 'data')],
    prevent_initial_call=True
)
def navigate_saliency(prev_clicks, next_clicks, current_index, saliency_data):
    """Handle saliency navigation button clicks."""
    global _current_pairs
    
    if not saliency_data or not _current_pairs:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'saliency-prev-btn' and current_index > 0:
        return current_index - 1
    elif button_id == 'saliency-next-btn' and current_index < len(_current_pairs) - 1:
        return current_index + 1
    
    return current_index


@callback(
    Output('saliency-current-index', 'data', allow_duplicate=True),
    Input('saliency-data', 'data'),
    prevent_initial_call=True
)
def reset_saliency_index_on_new_data(saliency_data):
    """Reset the saliency index to 0 when new saliency data is loaded."""
    if saliency_data and saliency_data.get('save_directory'):
        return 0
    raise PreventUpdate


# ------------------------------------------------------------
# New callback: switch saliency maps when an enhanced prompt is selected
# ------------------------------------------------------------

@callback(
    Output('saliency-data', 'data', allow_duplicate=True),
    Input('prompt-selection', 'value'),
    [State('cir-enhanced-prompts-data', 'data'), State('saliency-data', 'data')],
    prevent_initial_call=True
)
def switch_saliency_for_enhanced_prompt(selected_idx, enhanced_data, current_saliency):
    """Update saliency-data to display maps for the selected enhanced prompt (or revert to original)."""
    if enhanced_data is None:
        raise PreventUpdate

    # Determine directory to use
    if selected_idx is None or selected_idx == -1:
        dir_to_use = enhanced_data.get('initial_saliency_dir')
    else:
        dirs = enhanced_data.get('prompt_saliency_dirs', [])
        if 0 <= selected_idx < len(dirs):
            dir_to_use = dirs[selected_idx]
        else:
            dir_to_use = None

    # Fallback to original if none found
    if not dir_to_use:
        dir_to_use = enhanced_data.get('initial_saliency_dir')

    if not dir_to_use:
        raise PreventUpdate

    new_data = dict(current_saliency)
    new_data['save_directory'] = dir_to_use
    return new_data

# ------------------------------------------------------------
# Token Attribution display callback
# ------------------------------------------------------------

@callback(
    [Output('token-attribution-content', 'children', allow_duplicate=True),
     Output('token-attribution-navigation', 'style'),
     Output('token-attribution-current-info', 'children'),
     Output('ta-prev-btn', 'disabled'),
     Output('ta-next-btn', 'disabled')],
    [Input('saliency-data', 'data'),
     Input('token-attribution-index', 'data'),
     Input('cir-toggle-state', 'data'),
     Input('token-attr-fullscreen', 'data')],
    prevent_initial_call=True
)
def update_token_attribution_display(saliency_data, current_index, cir_toggle_state, token_attr_fullscreen):
    """Render token attribution barplot for the currently viewed candidate (no reference)."""
    global _current_pairs

    # Only show when visualization enabled
    if not cir_toggle_state:
        return (
            html.Div([
                html.I(className="fas fa-eye-slash text-muted me-2"),
                html.Span("Enable visualization to view token attribution", className="text-muted small")
            ], className="text-center p-2"),
            {'display':'none'}, "", True, True
        )

    # Validate data availability
    if not saliency_data or 'text_attribution' not in saliency_data:
        return (
            html.Div([
                html.I(className="fas fa-info-circle text-muted me-2"),
                 html.Span("Token attribution not available", className="text-muted")], className="p-2"),
            {'display':'none'}, "", True, True
        )

    text_attr = saliency_data.get('text_attribution', {})
    
    # Build list of ONLY candidate attributions (one per retrieved image)
    candidate_attributions = []

    candidate_attrs = text_attr.get('candidates', {})

    if candidate_attrs:
        # If we have rank-ordered pairs, use them to impose order
        if _current_pairs:
            for rank, _, _, pair_name in _current_pairs:
                # Keys in attribution dict may include extension; compare both raw and stem
                attr_data = None
                stem = Path(pair_name).stem
                for key, val in candidate_attrs.items():
                    if str(key) == str(pair_name) or str(key) == stem:
                        attr_data = val
                        break
                # Fallback: try substring match
                if attr_data is None:
                    for key, val in candidate_attrs.items():
                        if stem in str(key):
                            attr_data = val
                            break
                if attr_data is not None:
                    candidate_attributions.append((str(pair_name), attr_data, rank))
        
        # If some attributions were not matched (or no pairs) include remaining ones
        if len(candidate_attributions) == 0 or len(candidate_attributions) < len(candidate_attrs):
            seen = {a[0] for a in candidate_attributions}
            for idx, (key, val) in enumerate(candidate_attrs.items(), start=1):
                if str(key) in seen:
                    continue
                candidate_attributions.append((str(key), val, idx))
        
        # Sort by rank_num to keep stable order
        candidate_attributions.sort(key=lambda x: x[2])

    if not candidate_attributions:
        return (
            html.Div([
                html.I(className="fas fa-info-circle text-muted me-2"),
                html.Span("No candidate token attribution data found", className="text-muted")
            ], className="p-2"),
            {'display':'none'}, "", True, True
        )

    # Ensure current_index is within bounds
    if current_index < 0:
        current_index = 0
    elif current_index >= len(candidate_attributions):
        current_index = len(candidate_attributions) - 1

    # Get current attribution data
    attr_name, attr_data, rank_num = candidate_attributions[current_index]
    
    tokens = attr_data.get('tokens', [])
    attributions = attr_data.get('attributions', [])
    
    # Ensure data are JSON-serialisable lists
    try:
        attributions = [float(a) for a in attributions]
    except Exception:
        attributions = list(attributions)

    if not tokens or not attributions:
        return (
            html.Div([
                html.I(className="fas fa-info-circle text-muted me-2"),
                html.Span("Token attribution data empty", className="text-muted")
            ], className="p-2"),
            {'display':'none'}, "", True, True
        )

    # Resolve duplicate token labels by appending incremental suffix
    token_labels = []
    counts = {}
    for t in tokens:
        if t == '$':
            t = '[QUERY-IMAGE]'
        if t in counts:
            counts[t] += 1
            token_labels.append(f"{counts[t]*' '}{t}")
        else:
            counts[t] = 1
            token_labels.append(t)

    # Build Plotly bar chart
    fig = go.Figure(data=[
        go.Bar(x=token_labels, y=attributions, marker=dict(color=attributions, colorscale='Reds'))
    ])
    
    # Title for candidate attribution
    # title = f"Candidate Text Attribution (Rank {rank_num})"
    
    # Adjust chart size based on fullscreen mode
    if token_attr_fullscreen:
        chart_height = 400
        chart_margin = dict(l=30, r=30, t=50, b=100)
        title_font_size = 16
        axis_font_size = 12
    else:
        chart_height = 180
        chart_margin = dict(l=20, r=20, t=25, b=70)
        title_font_size = 11
        axis_font_size = 9
    
    fig.update_layout(
        height=chart_height,
        margin=chart_margin,
        xaxis_tickangle=-45,
        xaxis_title="Tokens",
        yaxis_title="Attribution Score",
        # title=title,
        title_font_size=title_font_size,
        xaxis_title_font_size=axis_font_size,
        yaxis_title_font_size=axis_font_size,
        template='simple_white'
    )

    graph = dcc.Graph(
        figure=fig, 
        config={'displayModeBar': False}, 
        style={'height': f'{chart_height}px', 'width': '100%'}
    )

    # Navigation info - show rank
    nav_info = f"Rank {rank_num}"

    # Navigation buttons state
    prev_dis = current_index <= 0
    next_dis = current_index >= len(candidate_attributions) - 1

    return (graph, {'display': 'block', 'padding': '0.25rem'}, nav_info, prev_dis, next_dis)


# ------------------------------------------------------------
# Navigation for Token Attribution
# ------------------------------------------------------------

@callback(
    Output('token-attribution-index', 'data'),
    [Input('ta-prev-btn', 'n_clicks'),
     Input('ta-next-btn', 'n_clicks')],
    [State('token-attribution-index', 'data'),
     State('saliency-data', 'data')],
    prevent_initial_call=True
)
def navigate_token_attribution(prev_clicks, next_clicks, current_index, saliency_data):
    """Handle prev/next navigation for token attribution charts."""
    if not saliency_data or 'text_attribution' not in saliency_data:
        raise PreventUpdate
    
    # Determine total candidates – prefer saliency pairs order when available
    text_attr = saliency_data.get('text_attribution', {})

    if _current_pairs:
        # Only count pairs that have matching attribution data
        cand_keys = set(text_attr.get('candidates', {}).keys())
        match_count = 0
        for _, _, _, pair_name in _current_pairs:
            stem = Path(pair_name).stem
            if pair_name in cand_keys or stem in cand_keys:
                match_count += 1
        total_candidates = match_count if match_count else len(text_attr.get('candidates', {}))
    else:
        total_candidates = len(text_attr.get('candidates', {}))
    
    if total_candidates == 0:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'ta-prev-btn' and current_index > 0:
        return current_index - 1
    elif button_id == 'ta-next-btn' and current_index < total_candidates - 1:
        return current_index + 1
    
    return current_index

@callback(
    Output('token-attribution-index', 'data', allow_duplicate=True),
    Input('saliency-data', 'data'),
    prevent_initial_call=True
)
def reset_token_attr_index_on_new_data(_):
    """Reset token-attribution index when new saliency-data arrives."""
    return 0 