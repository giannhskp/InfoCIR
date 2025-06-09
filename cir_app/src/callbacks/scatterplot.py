from dash import callback, Output, Input, State, dash, callback_context
from PIL import Image
from src import config
from src.Dataset import Dataset
from src.widgets import scatterplot
import plotly.graph_objects as go

@callback(
    Output('scatterplot', 'figure', allow_duplicate=True),
    State('scatterplot', 'figure'),
    Input('scatterplot', 'relayoutData'),
    prevent_initial_call=True,
)
def scatterplot_is_zoomed(scatterplot_fig, zoom_data):
    """Handle scatterplot zoom events to show images"""
    if len(zoom_data) == 1 and 'dragmode' in zoom_data:
        return dash.no_update

    if 'xaxis.range[0]' not in zoom_data:
        return dash.no_update

    print('Scatterplot is zoomed')
    return scatterplot.add_images_to_scatterplot(scatterplot_fig)

@callback(
    [Output("gallery", "children"),
    Output("wordcloud", "list"),
    Output('histogram', 'figure'),
    Output('scatterplot', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    State('scatterplot', 'figure'),
    Input("scatterplot", "selectedData"),
    Input('cir-toggle-state', 'data'),
    State('cir-search-data', 'data'),
    State('selected-gallery-image-ids', 'data'),
    prevent_initial_call=True,
)
def update_scatter_and_widgets(scatterplot_fig, selectedData, cir_toggle_state, search_data, selected_gallery_image_ids):
    """Unified handler for drag-select, visualize, and hide actions"""
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id']

    # Handle CIR toggle state changes
    if trigger.startswith('cir-toggle-state'):
        if not cir_toggle_state or not search_data:
            # Hide action: reset everything
            gallery_children = []
            wordcloud_data = []
            from src.widgets import histogram
            histogram_fig = histogram.draw_histogram(None)
            scatterplot_fig['layout']['images'] = []
            scatterplot_fig['data'][0]['marker'] = {'color': config.SCATTERPLOT_COLOR}
            scatterplot_fig['data'] = scatterplot_fig['data'][:3]
            return gallery_children, wordcloud_data, histogram_fig, scatterplot_fig, None, []
        else:
            # Visualize action
            from src.widgets import gallery, wordcloud, histogram
            df = Dataset.get()
            topk_ids = search_data.get('topk_ids', [])
            top1_id = search_data.get('top1_id', None)
            scatterplot_fig['data'] = scatterplot_fig['data'][:3]
            scatterplot_fig['layout']['images'] = []
            main_trace = scatterplot_fig['data'][0]
            xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
            axis_title = scatterplot_fig['layout']['xaxis']['title']['text']
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
            if xk:
                trace_k = go.Scatter(x=xk, y=yk, mode='markers', marker=dict(color=config.TOP_K_COLOR, size=7), name='Top-K')
                scatterplot_fig['data'].append(trace_k.to_plotly_json())
            if x1:
                trace_1 = go.Scatter(x=x1, y=y1, mode='markers', marker=dict(color=config.TOP_1_COLOR, size=9), name='Top-1')
                scatterplot_fig['data'].append(trace_1.to_plotly_json())
            # Only add Query trace for UMAP (not t-SNE, since t-SNE query position is just an approximation)
            if xq is not None and axis_title == 'umap_x':
                trace_q = go.Scatter(x=[xq], y=[yq], mode='markers', marker=dict(color=config.QUERY_COLOR, size=12, symbol='star'), name='Query')
                scatterplot_fig['data'].append(trace_q.to_plotly_json())
            # Add final composed query trace for UMAP
            if xfq is not None and axis_title == 'umap_x':
                trace_fq = go.Scatter(x=[xfq], y=[yfq], mode='markers', marker=dict(color=config.FINAL_QUERY_COLOR, size=10, symbol='diamond'), name='Final Query')
                scatterplot_fig['data'].append(trace_fq.to_plotly_json())
            class_counts = df.loc[topk_ids]['class_name'].value_counts()
            if len(class_counts):
                weights = wordcloud.wordcloud_weight_rescale(class_counts.values, 1, class_counts.max())
                wordcloud_data = sorted([[cn, w] for cn, w in zip(class_counts.index, weights)], key=lambda x: x[1], reverse=True)
            else:
                wordcloud_data = []
            
            # For CIR visualization, show ALL top-k images (not limited to IMAGE_GALLERY_SIZE)
            cir_data = df.loc[topk_ids]
            gallery_children = gallery.create_gallery_children(
                cir_data['image_path'].values, 
                cir_data['class_name'].values,
                cir_data.index.values,  # Pass image IDs for proper selection
                selected_gallery_image_ids  # Pass selected image IDs for highlighting
            )
            histogram_fig = histogram.draw_histogram(cir_data)
            return gallery_children, wordcloud_data, histogram_fig, scatterplot_fig, None, selected_gallery_image_ids

    # Drag selection action
    if trigger.startswith('scatterplot'):
        from src.widgets import gallery, wordcloud, histogram
        import numpy as np
        data_sel = scatterplot.get_data_selected_on_scatterplot(scatterplot_fig)
        scatterplot_fig['layout']['images'] = []
        class_counts = data_sel['class_name'].value_counts()
        if len(class_counts):
            weights = wordcloud.wordcloud_weight_rescale(class_counts.values, 1, class_counts.max())
            wordcloud_data = sorted([[cn, w] for cn, w in zip(class_counts.index, weights)], key=lambda x: x[1], reverse=True)
        else:
            wordcloud_data = []
        sample = data_sel.sample(min(len(data_sel), config.IMAGE_GALLERY_SIZE), random_state=1) if len(data_sel) else data_sel
        gallery_children = gallery.create_gallery_children(
            sample['image_path'].values, 
            sample['class_name'].values,
            sample.index.values,  # Pass image IDs for proper selection
            []  # Clear selected image IDs for new selections
        )
        histogram_fig = histogram.draw_histogram(data_sel)
        scatterplot.highlight_class_on_scatterplot(scatterplot_fig, None)
        return gallery_children, wordcloud_data, histogram_fig, scatterplot_fig, None, []

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update 