from dash import callback, Output, Input, State, dash
from PIL import Image
from src import config
from src.Dataset import Dataset
from src.widgets import scatterplot

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
    Output("gallery", "children"),
    Output("wordcloud", "list"),
    Output('histogram', 'figure'),
    Output('scatterplot', 'figure', allow_duplicate=True),
    State('scatterplot', 'figure'),
    Input("scatterplot", "selectedData"),
    prevent_initial_call=True,
)
def scatterplot_is_selected(scatterplot_fig, data_selected):
    """Handle scatterplot selection events"""
    print('Scatterplot is selected')

    from src.widgets import gallery, wordcloud, histogram
    import numpy as np

    data_selected = scatterplot.get_data_selected_on_scatterplot(scatterplot_fig)

    # Clear images from scatterplot
    scatterplot_fig['layout']['images'] = []

    # Calculate class distribution for selected data
    class_counts = data_selected['class_name'].value_counts()
    
    # Create wordcloud data
    if len(class_counts) > 0:
        wordcloud_weights = wordcloud.wordcloud_weight_rescale(
            class_counts.values,
            1,
            class_counts.max()
        )
        wordcloud_data = sorted(
            [[class_name, weight] for class_name, weight in zip(class_counts.index, wordcloud_weights)],
            key=lambda x: x[1], 
            reverse=True
        )
    else:
        wordcloud_data = []

    # Create gallery
    sample_data = data_selected.sample(
        min(len(data_selected), config.IMAGE_GALLERY_SIZE), 
        random_state=1
    ) if len(data_selected) > 0 else data_selected
    
    gallery_children = gallery.create_gallery_children(
        sample_data['image_path'].values, 
        sample_data['class_name'].values
    )

    # Create histogram
    histogram_fig = histogram.draw_histogram(data_selected)

    # Highlight selection on scatterplot
    scatterplot.highlight_class_on_scatterplot(scatterplot_fig, None)

    return gallery_children, wordcloud_data, histogram_fig, scatterplot_fig 