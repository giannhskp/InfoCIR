from dash import Input, Output, callback, State, dash
from src.widgets import scatterplot
import copy

@callback(
    [Output('scatterplot', 'figure', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('selected-histogram-class', 'data', allow_duplicate=True)],
    [State('scatterplot', 'figure'),
     State('histogram', 'figure'),
     State('selected-histogram-class', 'data')],
    Input("histogram", "clickData"),
    prevent_initial_call=True,
)
def histogram_is_clicked(scatterplot_fig, histogram_fig, selected_class, histogram_click):
    """Handle histogram bar clicks to toggle class highlight both on the scatterplot and within the histogram itself."""
    print('Histogram is clicked')

    # If clickData is None, do nothing
    if histogram_click is None or 'points' not in histogram_click or not histogram_click['points']:
        return dash.no_update, dash.no_update, dash.no_update

    # Retrieve full class name via customdata if present; fallback to x label
    point = histogram_click['points'][0]
    class_name = None
    if 'customdata' in point and point['customdata']:
        # customdata is a list with first element being class_name
        class_name = point['customdata'][0]
    if class_name is None:
        class_name = point.get('x')

    # Determine if we are toggling off (i.e., clicked the same class again)
    if selected_class == class_name:
        # Deselect – remove highlight
        new_selected_class = None
        scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [])
    else:
        # Select new class
        new_selected_class = class_name
        scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])

    # ------------------------------------------------------------------
    # Update histogram bar colours
    # ------------------------------------------------------------------
    new_hist_fig = copy.deepcopy(histogram_fig)

    if not new_hist_fig or 'data' not in new_hist_fig or not new_hist_fig['data']:
        # Figure is empty; nothing to update
        hist_update = dash.no_update
    else:
        bars = new_hist_fig['data'][0]
        # Get list of class names from customdata (original full names)
        class_names = [cd[0] if isinstance(cd, (list, tuple)) else cd for cd in bars.get('customdata', [])]
        default_color = 'rgba(31, 119, 180, 0.7)'
        from src import config
        bar_colors = [config.SELECTED_CLASS_COLOR if (new_selected_class and cn == new_selected_class) else default_color for cn in class_names]
        bars['marker']['color'] = bar_colors
        hist_update = new_hist_fig

    return scatterplot_fig, hist_update, new_selected_class 