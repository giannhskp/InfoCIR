import dash
from dash import callback, Output, Input, ALL, no_update, State
from dash.exceptions import PreventUpdate
from src.widgets import scatterplot

@callback(
    Output('scatterplot', 'figure', allow_duplicate=True),
    State('scatterplot', 'figure'),
    Input({'type': 'gallery-card', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True,
)
def gallery_image_is_clicked(scatterplot_fig, n_clicks):
    """Handle gallery image clicks to highlight class on scatterplot"""
    if all(e is None for e in n_clicks):
        return no_update

    print('Gallery is clicked')
    class_name = dash.callback_context.triggered_id['index']
    scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
    return scatterplot_fig 