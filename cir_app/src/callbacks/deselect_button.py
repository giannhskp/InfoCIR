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

    import copy

    # ------------------------------------------------------------------
    # 1. Reset the scatter-plot:
    #    • When visualisation is currently ON we want to *retain* the CIR
    #      traces and just clear user selections.
    #    • When visualisation is OFF we can safely rebuild the base figure
    #      the same way the old code did.
    # ------------------------------------------------------------------

    if cir_toggle_state:
        # --- Keep existing figure, just clear selections/overlays ---
        new_scatterplot_fig = copy.deepcopy(scatterplot_fig)

        # Clear box/lasso selections
        new_scatterplot_fig['layout']['selections'] = None

        # Remove any thumbnail images that might have been overlaid
        new_scatterplot_fig['layout']['images'] = []

        # Remove the helper trace used to highlight selected images (if present)
        new_scatterplot_fig['data'] = [
            trace for trace in new_scatterplot_fig['data']
            if trace.get('name') not in ['Selected Images', 'Selected Image']
        ]

        # Also clear the selectedpoints attribute of the main scatter trace (index 0)
        try:
            if 'selectedpoints' in new_scatterplot_fig['data'][0]:
                new_scatterplot_fig['data'][0]['selectedpoints'] = []
        except Exception:
            pass  # Robustness – keep going even if structure unexpected
    else:
        # --- Visualisation OFF -> return a clean base figure ---
        new_scatterplot_fig = scatterplot.create_scatterplot_figure(projection_selected)
        new_scatterplot_fig['layout'] = scatterplot_fig['layout']
        new_scatterplot_fig['layout']['selections'] = None

    # ------------------------------------------------------------------
    # 2. Update the *Visualize CIR results* toggle button.
    #    Requirement: if visualisation was ON before clicking "Deselect",
    #    auto-enable it again afterwards (i.e. keep it ON).  So we *retain*
    #    the previous ON state instead of turning it OFF.
    # ------------------------------------------------------------------
    if cir_toggle_state:
        # Visualisation was ON – keep it ON
        cir_button_text = 'Hide CIR results'
        cir_button_color = 'warning'
        new_cir_toggle_state = True
    else:
        # Visualisation was OFF – keep it OFF (unchanged behaviour)
        cir_button_text = 'Visualize CIR results'
        cir_button_color = 'success'
        new_cir_toggle_state = False

    # ------------------------------------------------------------------
    # 3. Build a blank histogram (since everything is deselected).
    # ------------------------------------------------------------------
    from src.widgets import histogram as histogram_widget
    histogram_fig = histogram_widget.draw_histogram(None)

    return new_scatterplot_fig, [], cir_button_text, cir_button_color, new_cir_toggle_state, -1, histogram_fig, None 