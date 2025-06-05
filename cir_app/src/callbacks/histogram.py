from dash import Input, Output, callback, State
from src.widgets import scatterplot

@callback(
    Output('scatterplot', 'figure', allow_duplicate=True),
    State('scatterplot', 'figure'),
    Input("histogram", "clickData"),
    prevent_initial_call=True,
)
def histogram_is_clicked(scatterplot_fig, histogram_click):
    """Handle histogram clicks to highlight class on scatterplot"""
    print('Histogram is clicked')
    if histogram_click is None:
        return scatterplot_fig
        
    class_name = histogram_click['points'][0]['x']
    scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
    return scatterplot_fig 