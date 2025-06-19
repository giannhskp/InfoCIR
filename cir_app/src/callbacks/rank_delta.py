from dash import callback, Input, Output, State, html, clientside_callback
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from src.Dataset import Dataset
import base64
import os


def _get_thumbnail_src(img_path: str, max_size: int = 60) -> str:
    """Return a base64 thumbnail for quick inline display."""
    if not img_path or not os.path.exists(img_path):
        return None
    try:
        from PIL import Image
        with Image.open(img_path) as im:
            im.thumbnail((max_size, max_size))
            import io
            buff = io.BytesIO()
            im.save(buff, format="JPEG", quality=85)
            return f"data:image/jpeg;base64,{base64.b64encode(buff.getvalue()).decode()}"
    except Exception:
        # Fallback to reading full image if PIL fails (rare)
        try:
            with open(img_path, "rb") as f:
                return f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Callback: build / update Rank-Δ table and highlight selected prompt row
# ---------------------------------------------------------------------------
# We listen to the ``prompt-selection`` RadioItems value to dynamically apply a
# highlight style to the corresponding prompt row when a user selects or
# deselects an enhanced prompt in the *Prompt enhancement* component.
@callback(
    Output("rank-delta-content", "children"),
    Input("cir-enhanced-prompts-data", "data"),          # newly generated / updated prompts
    Input("prompt-selection", "value"),                  # currently selected prompt index (or -1 / None)
    State("cir-search-data", "data"),
    prevent_initial_call=True,
)
def update_rank_delta_matrix(enhanced_data, selected_idx, search_data):
    """Compute and render the Rank-Δ matrix once enhanced prompts are available."""
    if not enhanced_data or not search_data:
        return html.Div([
            html.I(className="fas fa-table text-muted me-2"),
            "Run prompt enhancement to view the rank-Δ matrix."
        ], className="text-muted p-4 text-center")

    prompts = enhanced_data.get("prompts")
    all_results = enhanced_data.get("all_results")
    original_ids = search_data.get("topk_ids")
    best_idx = enhanced_data.get("best_idx", 0)
    # ``selected_idx`` may come through as None; normalise to -1 for easier checks
    if selected_idx is None:
        selected_idx = -1

    if not prompts or not all_results or not original_ids:
        return html.Div([
            html.I(className="fas fa-exclamation-triangle text-warning me-2"),
            "Insufficient data to compute Rank-Δ matrix."
        ], className="text-warning p-4 text-center")

    # Convert all IDs to strings for consistent comparison
    original_ids_str = [str(x) for x in original_ids]
    df = Dataset.get()

    # Create improved header row with prompt IDs instead of full text
    header_cells = [html.Th([
        html.I(className="fas fa-image me-1", style={"fontSize": "0.7rem"}),
        "Images"
    ], className="bg-light fw-bold text-center", style={
        "verticalAlign": "middle", 
        "fontSize": "0.75rem",
        "padding": "0.5rem",
        "borderRight": "2px solid #dee2e6"
    })]
    
    # Add image thumbnail headers (limit to ensure visibility)
    max_images = min(10, len(original_ids_str))  # Show max 10 images to ensure all columns are visible
    for i, img_id in enumerate(original_ids_str[:max_images]):
        img_path = None
        # Dataset may have numeric or string index
        if img_id in df.index:
            img_path = df.loc[img_id]["image_path"]
        else:
            try:
                idx = int(img_id)
                if idx in df.index:
                    img_path = df.loc[idx]["image_path"]
            except Exception:
                pass
        
        thumb_src = _get_thumbnail_src(img_path, max_size=35) if img_path else None
        header_content = html.Div([
            html.Img(src=thumb_src, style={
                "height": "30px", 
                "width": "30px", 
                "objectFit": "cover", 
                "borderRadius": "4px",
                "border": "1px solid #dee2e6"
            }) if thumb_src else html.I(className="fas fa-image text-muted"),
            html.Div(f"#{i+1}", className="small text-muted mt-1", style={"fontSize": "0.6rem"})
        ], className="d-flex flex-column align-items-center")
        
        header_cells.append(html.Th(header_content, 
                                   className="bg-light text-center", 
                                   style={
                                       "padding": "0.4rem", 
                                       "minWidth": "50px",
                                       "fontSize": "0.7rem"
                                   }))

    # Build matrix rows with prompt IDs and hover tooltips
    body_rows = []
    for prompt_idx, (prompt_text, prompt_results) in enumerate(zip(prompts, all_results)):
        result_ids = [str(name) for name, _ in prompt_results]
        is_best = (prompt_idx == best_idx)
        is_selected = (prompt_idx == selected_idx)
        
        # Create prompt ID cell with tooltip on hover
        prompt_cell_style = {
            "whiteSpace": "nowrap", 
            "padding": "0.5rem",
            "fontSize": "0.75rem",
            "cursor": "help",
            "borderRight": "2px solid #dee2e6"
        }
        
        if is_best:
            prompt_cell_content = html.Div([
                html.I(className="fas fa-crown text-warning me-1", style={"fontSize": "0.6rem"}),
                f"P{prompt_idx + 1}"
            ], className="fw-bold")
        else:
            prompt_cell_content = html.Div([
                html.I(className="fas fa-magic text-info me-1", style={"fontSize": "0.6rem"}),
                f"P{prompt_idx + 1}"
            ])
        
        # Create a prettier tooltip using custom CSS
        tooltip_content = html.Div([
            prompt_cell_content,
            html.Div([
                html.Div(f'Prompt {prompt_idx + 1}', className="tooltip-title"),
                html.Div(f'"{prompt_text}"', className="tooltip-text")
            ], className="custom-tooltip")
        ], className="tooltip-container")
        
        row_cells = [html.Td(tooltip_content, style=prompt_cell_style)]
        
        # Add delta cells for each image (limited to max_images)
        for orig_rank, img_id in enumerate(original_ids_str[:max_images]):
            if img_id in result_ids:
                new_rank = result_ids.index(img_id)  # 0-based
                delta = new_rank - orig_rank
                
                if delta == 0:
                    cell_value = "="
                    badge_class = "badge bg-info bg-opacity-75"
                elif delta < 0:
                    cell_value = str(delta)  # Already negative
                    badge_class = "badge bg-success bg-opacity-75"
                else:
                    cell_value = f"+{delta}"
                    badge_class = "badge bg-danger bg-opacity-75"
            else:
                cell_value = "out"
                badge_class = "badge bg-secondary bg-opacity-75"
            
            cell_content = html.Span(cell_value, className=badge_class, style={"fontSize": "0.7rem"})
            row_cells.append(html.Td(cell_content, 
                                   className="text-center align-middle",
                                   style={"padding": "0.4rem"}))
        
        # ------------------------------------------------------------------
        # Determine row CSS classes
        # ------------------------------------------------------------------
        if is_selected:
            # Blue-ish highlight for the currently selected prompt
            row_class = "table-info bg-opacity-25"
        elif is_best:
            # Green highlight for best prompt (if not currently selected)
            row_class = "table-success bg-opacity-25"
        else:
            row_class = ""

        # Each row gets an explicit ID so that we can scroll to it from a
        # client-side callback.
        body_rows.append(html.Tr(row_cells, className=row_class, id=f"rank-row-{prompt_idx}"))

    # Create styled table
    rank_delta_table = dbc.Table([
        html.Thead(html.Tr(header_cells), className="thead-light"),
        html.Tbody(body_rows),
    ], 
    bordered=True, 
    hover=True, 
    responsive=True,
    className="rank-delta-table mb-3",
    style={
        "fontSize": "0.8rem",
        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
        "borderRadius": "6px",
        "overflow": "hidden"
    })

    # Compact and attractive legend
    legend = html.Div([
        html.Div([
            html.I(className="fas fa-info-circle text-primary me-2", style={"fontSize": "0.8rem"}),
            html.Strong("Legend: ", style={"fontSize": "0.75rem"}),
            html.Span([
                html.Span("−", className="badge bg-success bg-opacity-75 me-1", style={"fontSize": "0.6rem"}),
                html.Span("moved up ", style={"fontSize": "0.7rem", "color": "#28a745"}),
                html.Span("+", className="badge bg-danger bg-opacity-75 me-1", style={"fontSize": "0.6rem"}),
                html.Span("moved down ", style={"fontSize": "0.7rem", "color": "#dc3545"}),
                html.Span("=", className="badge bg-info bg-opacity-75 me-1", style={"fontSize": "0.6rem"}),
                html.Span("same rank ", style={"fontSize": "0.7rem"}),
                html.Span("out", className="badge bg-secondary bg-opacity-75", style={"fontSize": "0.6rem"}),
                html.Span(" not in top-k", style={"fontSize": "0.7rem"})
            ])
        ], className="d-flex align-items-center flex-wrap justify-content-center")
    ], className="p-2 mb-3 bg-light border-start border-primary border-4 rounded")

    # Instruction text
    instruction = html.Div([
        html.I(className="fas fa-lightbulb text-info me-1", style={"fontSize": "0.7rem"}),
        html.Small([
            "Hover over ", 
            html.Strong("P1, P2, etc."), 
            " to see full prompts"
        ], className="text-muted", style={"fontSize": "0.7rem"})
    ], className="text-center mt-3")

    return html.Div([
        # Table container with horizontal scroll
        html.Div(rank_delta_table, style={"overflowX": "auto"}),
        
        legend,
        instruction,
    ], style={"padding": "0.5rem"}) 