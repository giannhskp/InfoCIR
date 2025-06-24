import dash_bootstrap_components as dbc
from src import config

def create_help_popup():
    """Create help popup modal"""
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("How to use")),
            dbc.ModalBody('With this tool you can explore your dataset through a 2D UMAP projection of the CLIP embeddings of the images.'),
            dbc.ModalBody('Use the scatterplot to select instances of images. '
                          'Click on the scatterplot icons above the scatterplot to select mode of use. '
                          'Double click on scatterplot selections while using the select tool to deselect.'),
            dbc.ModalBody(f'Use the widgets in the tabs to explore the data. The gallery shows you a sample of up to {config.IMAGE_GALLERY_SIZE} images. '
                          'The wordcloud and histogram show the most prevalent classes in a selection. '),
        ],
        id="help-popup",
        is_open=False,
    ) 