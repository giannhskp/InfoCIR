from dash import Dash, html, dcc
from src import config
from src.Dataset import Dataset
from src.widgets import (
    projection_radio_buttons, 
    gallery, 
    scatterplot, 
    wordcloud, 
    histogram, 
    help_popup
)
import dash_bootstrap_components as dbc

# Import callbacks
import src.callbacks.scatterplot
import src.callbacks.projection_radio_buttons
import src.callbacks.wordcloud
import src.callbacks.histogram
import src.callbacks.gallery
import src.callbacks.deselect_button
import src.callbacks.help_button

def run_ui():
    """Run the Dash UI application"""
    external_stylesheets = [dbc.themes.BOOTSTRAP]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    
    # Create widgets
    help_popup_widget = help_popup.create_help_popup()
    projection_radio_buttons_widget = projection_radio_buttons.create_projection_radio_buttons()
    scatterplot_widget = scatterplot.create_scatterplot(config.DEFAULT_PROJECTION)
    wordcloud_widget = wordcloud.create_wordcloud()
    gallery_widget = gallery.create_gallery()
    histogram_widget = histogram.create_histogram()

    # Create right tab with available widgets
    right_tab = dcc.Tabs([
        dcc.Tab(label='wordcloud', children=wordcloud_widget),
        dcc.Tab(label='images', children=gallery_widget),
        dcc.Tab(label='histogram', children=histogram_widget),
    ])

    # Create app layout
    app.layout = dbc.Container([
        help_popup_widget,
        dbc.Stack([
            projection_radio_buttons_widget,
            dbc.Button('Deselect everything', 
                      id='deselect-button', 
                      class_name="btn btn-outline-primary ms-auto header-button"),
            dbc.Button('Help', 
                      id='help-button', 
                      class_name="btn btn-outline-primary header-button")
        ], id='header', direction="horizontal"),
        dbc.Row([
            dbc.Col(scatterplot_widget, width=6, className='main-col'),
            dbc.Col(right_tab, width=6, className='main-col')
        ], className='h-100', justify='between')
    ], fluid=True, id='container')

    app.run(debug=True, use_reloader=False, port=config.PORT)

def main():
    """Main function to initialize dataset and start the app"""
    # Check if we have processed data already
    if not Dataset.processed_files_exist():
        print('Processed dataset not found. Creating from source data...')
        Dataset.download()

    # Load the processed dataset
    Dataset.load()

    # Check if sample size changed and we need to reprocess
    if len(Dataset.get()) != config.DATASET_SAMPLE_SIZE:
        print('Sample size changed in the configuration. Recalculating features.')
        Dataset.download()
        Dataset.load()

    print('Starting Dash application')
    run_ui()

if __name__ == '__main__':
    main() 