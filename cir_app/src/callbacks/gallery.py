import dash
from dash import callback, Output, Input, ALL, no_update, State
from dash.exceptions import PreventUpdate
from src.widgets import scatterplot

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data')],
    [State('scatterplot', 'figure'),
     State('selected-image-data', 'data')],
    Input({'type': 'gallery-card', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def gallery_image_is_clicked(scatterplot_fig, selected_image_data, n_clicks):
    """Handle gallery image clicks to highlight class on scatterplot"""
    if all(e is None for e in n_clicks):
        return no_update, no_update

    print('Gallery is clicked')
    triggered_id = dash.callback_context.triggered_id['index']
    
    try:
        # Parse the new string format
        if triggered_id.startswith("image_"):
            # Format: "image_{image_id}_{class_name}"
            parts = triggered_id.split("_", 2)  # Split into at most 3 parts
            if len(parts) >= 3:
                image_id = int(parts[1])
                class_name = parts[2]
                print(f"Selected image ID: {image_id}, class: {class_name}")
                
                # Handle restoration of previous selected image if CIR is active
                has_cir_traces = any(trace.get('name') in ['Top-K', 'Top-1', 'Query', 'Final Query'] 
                                   for trace in scatterplot_fig['data'])
                
                if has_cir_traces and selected_image_data:
                    # Restore previous selected image to its CIR trace
                    _restore_previous_selected_image(scatterplot_fig, selected_image_data)
                
                # Highlight the specific selected image and its class
                scatterplot.highlight_selected_image_and_class(scatterplot_fig, image_id, [class_name])
                
                # Store new selected image info if CIR is active
                new_selected_data = None
                if has_cir_traces:
                    new_selected_data = _create_selected_image_data(scatterplot_fig, image_id)
                
                return scatterplot_fig, new_selected_data
            else:
                raise ValueError(f"Invalid image identifier format: {triggered_id}")
        elif triggered_id.startswith("class_"):
            # Format: "class_{class_name}" (backwards compatibility)
            class_name = triggered_id[6:]  # Remove "class_" prefix
            print(f"Selected class: {class_name}")
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
            return scatterplot_fig, None
        else:
            # Legacy format: just the class name directly
            class_name = triggered_id
            print(f"Selected class (legacy): {class_name}")
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
            return scatterplot_fig, None
            
    except Exception as e:
        print(f"Error in gallery callback: {e}")
        # Fallback: try to extract class name and highlight just the class
        if triggered_id.startswith("image_"):
            parts = triggered_id.split("_", 2)
            if len(parts) >= 3:
                class_name = parts[2]
                scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
        elif triggered_id.startswith("class_"):
            class_name = triggered_id[6:]
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
        else:
            scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [str(triggered_id)])
    
    return scatterplot_fig, no_update


def _restore_previous_selected_image(scatterplot_fig, selected_image_data):
    """Restore previous selected image to its CIR trace"""
    if not selected_image_data:
        return
    
    prev_x = selected_image_data['x']
    prev_y = selected_image_data['y']
    prev_was_top1 = selected_image_data['was_top1']
    
    # Find CIR traces and restore the previous selected image
    for trace in scatterplot_fig['data']:
        if prev_was_top1 and trace.get('name') == 'Top-1':
            trace['x'] = [prev_x]
            trace['y'] = [prev_y]
            break
        elif not prev_was_top1 and trace.get('name') == 'Top-K':
            if 'x' not in trace or trace['x'] is None:
                trace['x'] = []
                trace['y'] = []
            trace['x'].append(prev_x)
            trace['y'].append(prev_y)
            break


def _create_selected_image_data(scatterplot_fig, image_id):
    """Create selected image data from the current scatterplot"""
    from src.Dataset import Dataset
    
    # Find main data trace
    main_trace = scatterplot_fig['data'][0]
    xs, ys, cds = main_trace['x'], main_trace['y'], main_trace['customdata']
    
    # Find coordinates for the selected image
    selected_x, selected_y = None, None
    for xi, yi, idx in zip(xs, ys, cds):
        if int(idx) == image_id:
            selected_x, selected_y = xi, yi
            break
    
    if selected_x is None:
        return None
    
    # Check if this image was in Top-1 trace (before it was moved to Selected Image trace)
    # We'll determine this by checking if there's a Selected Image trace and no Top-1 trace with data
    selected_was_top1 = False
    top1_trace_empty = False
    
    for trace in scatterplot_fig['data']:
        if trace.get('name') == 'Top-1':
            if not trace.get('x') or len(trace['x']) == 0:
                top1_trace_empty = True
        elif trace.get('name') == 'Selected Image':
            # If Selected Image trace exists and Top-1 is empty, selected image was top-1
            if top1_trace_empty:
                selected_was_top1 = True
    
    return {
        'image_id': image_id,
        'x': selected_x,
        'y': selected_y,
        'was_top1': selected_was_top1
    } 