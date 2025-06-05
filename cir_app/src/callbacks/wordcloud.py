from dash import Input, Output, callback, State
from src.widgets import scatterplot

@callback(
    Output('scatterplot', 'figure', allow_duplicate=True),
    State('scatterplot', 'figure'),
    Input("wordcloud", "click"),
    prevent_initial_call=True,
)
def wordcloud_is_clicked(scatterplot_fig, wordcloud_selection):
    """Handle wordcloud clicks to highlight class on scatterplot"""
    print('Wordcloud is clicked')
    if wordcloud_selection is None:
        return scatterplot_fig
        
    class_name = wordcloud_selection[0]
    scatterplot.highlight_class_on_scatterplot(scatterplot_fig, [class_name])
    return scatterplot_fig 