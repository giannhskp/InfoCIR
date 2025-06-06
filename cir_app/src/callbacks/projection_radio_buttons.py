from dash import callback, Output, Input, State
from src.widgets import scatterplot
from src.Dataset import Dataset
from src import config
import plotly.graph_objects as go

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('cir-visualize-button', 'disabled', allow_duplicate=True),
     Output('cir-hide-button', 'disabled', allow_duplicate=True)],
    Input('projection-radio-buttons', 'value'),
    [State('cir-hide-button', 'disabled'),
     State('cir-search-data', 'data')],
    prevent_initial_call=True,
)
def projection_radio_is_clicked(radio_button_value, hide_button_disabled, search_data):
    """Handle projection button changes and preserve CIR visualization state"""
    print('Radio button is clicked')
    
    # Create the new scatterplot
    new_fig = scatterplot.create_scatterplot_figure(radio_button_value)
    
    # Check if CIR results are currently visualized (hide button enabled = visualization active)
    cir_is_visualized = not hide_button_disabled and search_data is not None
    
    if cir_is_visualized:
        # Re-apply CIR visualization to the new scatterplot
        df = Dataset.get()
        topk_ids = search_data.get('topk_ids', [])
        top1_id = search_data.get('top1_id', None)
        
        # Get coordinates based on the new projection
        if radio_button_value == 'UMAP':
            xq, yq = search_data['umap_x_query'], search_data['umap_y_query']
        else:  # t-SNE
            xq, yq = search_data['tsne_x_query'], search_data['tsne_y_query']
        
        # Extract data from the new scatterplot
        main_trace = new_fig['data'][0]
        xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
        
        # Find coordinates for top-k and top-1
        x1, y1, xk, yk = [], [], [], []
        top1_id_cmp = int(top1_id) if top1_id is not None else None
        topk_ids_cmp = [int(x) for x in topk_ids]
        
        for xi, yi, idx in zip(xs, ys, cds):
            idx_cmp = int(idx) if idx is not None else None
            if idx_cmp == top1_id_cmp:
                x1.append(xi); y1.append(yi)
            elif idx_cmp in topk_ids_cmp:
                xk.append(xi); yk.append(yi)
        
        # Add CIR traces to the figure
        if xk:
            trace_k = go.Scatter(x=xk, y=yk, mode='markers', marker=dict(color=config.TOP_K_COLOR, size=7), name='Top-K')
            new_fig.add_trace(trace_k)
        if x1:
            trace_1 = go.Scatter(x=x1, y=y1, mode='markers', marker=dict(color=config.TOP_1_COLOR, size=9), name='Top-1')
            new_fig.add_trace(trace_1)
        if xq is not None and radio_button_value == 'UMAP':
            trace_q = go.Scatter(x=[xq], y=[yq], mode='markers', marker=dict(color=config.QUERY_COLOR, size=12, symbol='star'), name='Query')
            new_fig.add_trace(trace_q)
        
        # Keep current button states (visualization remains active)
        return new_fig, True, False
    else:
        # No CIR visualization active, reset button states
        return new_fig, False, True 