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
     Input('cir-toggle-state', 'data')],
    prevent_initial_call=True
)
def update_saliency_display(saliency_data, current_index, cir_toggle_state):
    """Update the saliency display based on current data and index."""
    global _current_pairs
    
    # Only show saliency when CIR results are being visualized
    if not cir_toggle_state:
        return ([
            html.Div([
                html.I(className="fas fa-brain text-info me-2"),
                html.H5("Saliency Maps", className="d-inline"),
                html.P("Enable 'Visualize CIR results' to view saliency maps.", 
                       className="text-muted mt-2")
            ], className="text-center p-4")
        ], {'display': 'none'}, "", True, True)
    
    if not saliency_data or not saliency_data.get('save_directory'):
        return ([
            html.Div([
                html.I(className="fas fa-brain text-info me-2"),
                html.H5("Saliency Maps", className="d-inline"),
                html.P("No saliency data available. Run a CIR query with SEARLE to generate saliency maps.", 
                       className="text-muted mt-2")
            ], className="text-center p-4")
        ], {'display': 'none'}, "", True, True)
    
    # Use cached pairs
    pairs = _current_pairs
    
    if not pairs:
        return ([
            html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.H5("No Saliency Maps Found", className="d-inline"),
                html.P("Saliency files were not found in the expected location.", 
                       className="text-muted mt-2"),
                html.Small(f"Searched in: {saliency_data['save_directory']}", 
                          className="text-muted")
            ], className="text-center p-4")
        ], {'display': 'none'}, "", True, True)
    
    # Ensure current_index is within bounds
    if current_index < 0:
        current_index = 0
    elif current_index >= len(pairs):
        current_index = len(pairs) - 1
    
    # Get current pair
    rank, candidate_path, reference_path, image_name = pairs[current_index]
    
    # Load images with optimization
    try:
        reference_src = load_and_resize_image(reference_path)
        candidate_src = load_and_resize_image(candidate_path)
        
        if not reference_src or not candidate_src:
            # If images fail to load, show error
            content = [
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                    html.H5("Error Loading Images", className="d-inline"),
                    html.P("Could not load saliency images from disk.", 
                           className="text-muted mt-2")
                ], className="text-center p-4")
            ]
        else:
            # Create display content
            content = [
                # Header
                html.Div([
                    html.H5([
                        html.I(className="fas fa-brain text-info me-2"),
                        f"Saliency Analysis - Rank {rank}"
                    ], className="mb-3"),
                    html.P([
                        "Showing paired saliency maps for ",
                        html.Code(image_name, className="bg-light px-1"),
                        ". The reference saliency shows which parts of the reference image support this candidate match."
                    ], className="text-muted small mb-4")
                ]),
                
                # Image pair
                html.Div([
                    dbc.Row([
                        # Reference saliency (similarity-based)
                        dbc.Col([
                            html.Div([
                                html.H6([
                                    html.I(className="fas fa-search me-1 text-primary"),
                                    "Reference Saliency"
                                ], className="text-primary mb-2"),
                                html.P("Which parts of the reference support this match", 
                                       className="small text-muted mb-3"),
                                html.Div([
                                    html.Img(
                                        src=reference_src,
                                        style={
                                            'width': '100%',
                                            'height': 'auto',
                                            'maxHeight': '350px',
                                            'objectFit': 'contain'
                                        }
                                    )
                                ], className="saliency-image-container", 
                                   style={
                                       'border': '2px solid #0d6efd',
                                       'borderRadius': '8px',
                                       'padding': '4px',
                                       'backgroundColor': '#f8f9fa'
                                   })
                            ])
                        ], width=6, className="mb-3"),
                        
                        # Candidate saliency
                        dbc.Col([
                            html.Div([
                                html.H6([
                                    html.I(className="fas fa-eye me-1 text-success"),
                                    "Candidate Saliency"
                                ], className="text-success mb-2"),
                                html.P("Which parts of this candidate drive the similarity", 
                                       className="small text-muted mb-3"),
                                html.Div([
                                    html.Img(
                                        src=candidate_src,
                                        style={
                                            'width': '100%',
                                            'height': 'auto',
                                            'maxHeight': '350px',
                                            'objectFit': 'contain'
                                        }
                                    )
                                ], className="saliency-image-container",
                                   style={
                                       'border': '2px solid #198754',
                                       'borderRadius': '8px',
                                       'padding': '4px',
                                       'backgroundColor': '#f8f9fa'
                                   })
                            ])
                        ], width=6, className="mb-3")
                    ], className="g-3")
                ], className="saliency-pair-container")
            ]
            
    except Exception as e:
        print(f"Error processing saliency images: {e}")
        content = [
            html.Div([
                html.I(className="fas fa-exclamation-triangle text-warning me-2"),
                html.H5("Error Processing Images", className="d-inline"),
                html.P(f"Error: {str(e)}", className="text-muted mt-2")
            ], className="text-center p-4")
        ]
    
    # Navigation info
    nav_info = f"Showing {current_index + 1} of {len(pairs)}"
    
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

    # When directory changes, reset current index to 0 via preprocessing chain
    return {'save_directory': dir_to_use} 