from dash import callback, Input, Output, State, html
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


@callback(
    Output("rank-delta-content", "children"),
    Input("cir-enhanced-prompts-data", "data"),
    State("cir-search-data", "data"),
    prevent_initial_call=True,
)
def update_rank_delta_matrix(enhanced_data, search_data):
    """Compute and render the Rank-Δ matrix once enhanced prompts are available."""
    if not enhanced_data or not search_data:
        raise PreventUpdate

    prompts = enhanced_data.get("prompts")
    all_results = enhanced_data.get("all_results")
    original_ids = search_data.get("topk_ids")

    if not prompts or not all_results or not original_ids:
        return html.Div("Insufficient data to compute Rank-Δ matrix.", className="text-danger p-4")

    # Convert all IDs to strings for consistent comparison
    original_ids_str = [str(x) for x in original_ids]

    df = Dataset.get()

    # Header row – first blank corner cell, then thumbnails
    header_cells = [html.Th("Prompt / Image", style={"verticalAlign": "bottom"})]
    for img_id in original_ids_str:
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
        thumb_src = _get_thumbnail_src(img_path) if img_path else None
        if thumb_src:
            header_cells.append(html.Th(html.Img(src=thumb_src, style={"height": "40px"}),
                                        style={"textAlign": "center"}))
        else:
            header_cells.append(html.Th(str(img_id), style={"fontSize": "12px", "textAlign": "center"}))

    # Build matrix rows
    body_rows = []
    for prompt_text, prompt_results in zip(prompts, all_results):
        result_ids = [str(name) for name, _ in prompt_results]
        row_cells = [html.Td(prompt_text, style={"whiteSpace": "normal", "maxWidth": "220px"})]
        for orig_rank, img_id in enumerate(original_ids_str):
            if img_id in result_ids:
                new_rank = result_ids.index(img_id)  # 0-based
                delta = new_rank - orig_rank
                cell_value = f"{delta:+d}" if delta != 0 else "0"
                # Basic colouring: green for negative, red for positive, white for zero
                if delta < 0:
                    color = "#d4edda"  # light green
                elif delta > 0:
                    color = "#f8d7da"  # light red
                else:
                    color = "#ffffff"
            else:
                cell_value = "out"
                color = "#fff3cd"  # light yellow
            row_cells.append(html.Td(cell_value, style={"textAlign": "center", "backgroundColor": color}))
        body_rows.append(html.Tr(row_cells))

    rank_delta_table = html.Table([
        html.Thead(html.Tr(header_cells)),
        html.Tbody(body_rows),
    ], className="table table-bordered table-sm text-center")

    caption = html.P(
        "Rank-Δ values (negative = moved up, positive = moved down, 'out' = not in results)",
        className="small text-muted",
    )

    return html.Div([caption, rank_delta_table], style={"overflowX": "auto", "padding": "1rem"}) 