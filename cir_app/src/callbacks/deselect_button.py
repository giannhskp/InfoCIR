from dash import Input, Output, callback, State
import plotly.graph_objects as go
from src.widgets import scatterplot

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True),
     Output('cir-toggle-button', 'children', allow_duplicate=True),
     Output('cir-toggle-button', 'color', allow_duplicate=True),
     Output('cir-toggle-state', 'data', allow_duplicate=True),
     Output('prompt-selection', 'value', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('selected-histogram-class', 'data', allow_duplicate=True)],
    [State('projection-radio-buttons', 'value'),
     State('scatterplot', 'figure'),
     State('cir-toggle-state', 'data')],
    Input('deselect-button', 'n_clicks'),
    prevent_initial_call=True,
)
def deselect_button_is_pressed(projection_selected, scatterplot_fig, cir_toggle_state, _):
    """Handle deselect button clicks"""
    print('Deselect button is clicked')
    
    # Create new scatterplot figure without selections
    new_scatterplot_fig = scatterplot.create_scatterplot_figure(projection_selected)
    new_scatterplot_fig['layout'] = scatterplot_fig['layout']
    new_scatterplot_fig['layout']['selections'] = None
    
    # If CIR results are currently visualized, hide them
    if cir_toggle_state:
        # Auto-hide CIR results by setting toggle state to False
        cir_button_text = 'Visualize CIR results'
        cir_button_color = 'success'
        new_cir_toggle_state = False
    else:
        # Keep current CIR button state
        cir_button_text = 'Visualize CIR results' if not cir_toggle_state else 'Hide CIR results'
        cir_button_color = 'success' if not cir_toggle_state else 'warning'
        new_cir_toggle_state = cir_toggle_state
    
    # Build a blank histogram (no selection)
    from src.widgets import histogram as histogram_widget
    histogram_fig = histogram_widget.draw_histogram(None)

    return new_scatterplot_fig, [], cir_button_text, cir_button_color, new_cir_toggle_state, -1, histogram_fig, None 