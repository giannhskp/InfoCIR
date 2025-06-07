from dash import callback, Output, Input, State
from src.widgets import scatterplot
from src.Dataset import Dataset
from src import config
import plotly.graph_objects as go

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('cir-visualize-button', 'disabled', allow_duplicate=True),
     Output('cir-hide-button', 'disabled', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('projection-radio-buttons', 'value'),
    [State('cir-hide-button', 'disabled'),
     State('cir-search-data', 'data')],
    prevent_initial_call=True,
)
def projection_radio_is_clicked(radio_button_value, hide_button_disabled, search_data):
    """Handle projection radio button clicks"""
    print('Projection radio button is clicked')
    new_scatterplot_fig = scatterplot.create_scatterplot_figure(radio_button_value)
    
    # If CIR is active and visible, re-add the visualization traces
    if search_data and not hide_button_disabled:  # hide_button_disabled=False means CIR is visible
        df = Dataset.get()
        topk_ids = search_data.get('topk_ids', [])
        top1_id = search_data.get('top1_id', None)
        
        # Get trace data for the new projection
        main_trace = new_scatterplot_fig['data'][0]
        xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
        
        # Get query coordinates based on projection type
        axis_title = new_scatterplot_fig['layout']['xaxis']['title']['text']
        if axis_title == 'umap_x':
            xq, yq = search_data['umap_x_query'], search_data['umap_y_query']
            xfq, yfq = search_data.get('umap_x_final_query'), search_data.get('umap_y_final_query')
        else:
            xq, yq = search_data['tsne_x_query'], search_data['tsne_y_query']
            xfq, yfq = None, None  # Final query only shown for UMAP
        
        x1, y1, xk, yk = [], [], [], []
        
        # Ensure consistent types for comparison
        top1_id_cmp = int(top1_id) if top1_id is not None else None
        topk_ids_cmp = [int(x) for x in topk_ids]
        
        for xi, yi, idx in zip(xs, ys, cds):
            idx_cmp = int(idx) if idx is not None else None
            if idx_cmp == top1_id_cmp:
                x1.append(xi); y1.append(yi)
            elif idx_cmp in topk_ids_cmp:
                xk.append(xi); yk.append(yi)
        
        # Add CIR traces
        if xk:
            trace_k = go.Scatter(x=xk, y=yk, mode='markers', marker=dict(color=config.TOP_K_COLOR, size=7), name='Top-K')
            new_scatterplot_fig['data'].append(trace_k.to_plotly_json())
        if x1:
            trace_1 = go.Scatter(x=x1, y=y1, mode='markers', marker=dict(color=config.TOP_1_COLOR, size=9), name='Top-1')
            new_scatterplot_fig['data'].append(trace_1.to_plotly_json())
        # Only add Query trace for UMAP
        if xq is not None and axis_title == 'umap_x':
            trace_q = go.Scatter(x=[xq], y=[yq], mode='markers', marker=dict(color=config.QUERY_COLOR, size=12, symbol='star'), name='Query')
            new_scatterplot_fig['data'].append(trace_q.to_plotly_json())
        # Add final composed query trace for UMAP
        if xfq is not None and axis_title == 'umap_x':
            trace_fq = go.Scatter(x=[xfq], y=[yfq], mode='markers', marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond'), name='Final Query')
            new_scatterplot_fig['data'].append(trace_fq.to_plotly_json())
    
    # Clear selected image data and gallery highlighting when changing projections
    return new_scatterplot_fig, False, True, None, [] 