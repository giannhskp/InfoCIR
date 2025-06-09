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
import src.callbacks.cir_callbacks

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

    # Create CIR interface
    cir_interface = dbc.Card([
        dbc.CardHeader(html.H4("Composed Image Retrieval", className="mb-0")),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Upload Query Image:", className="form-label fw-bold"),
                    dcc.Upload(
                        id='cir-upload-image',
                        children=html.Div([
                            html.I(className="fas fa-cloud-upload-alt me-2"),
                            'Drag and Drop or Click to Select Image'
                        ]),
                        style={
                            'width': '100%',
                            'height': '80px',
                            'lineHeight': '80px',
                            'borderWidth': '2px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'background': '#fafafa',
                            'cursor': 'pointer',
                            'color': '#666'
                        },
                        className='cir-upload-area',
                        multiple=False,
                        accept='image/*'
                    ),
                    html.Div(id='cir-upload-status', className="mt-2 status-indicator")
                ], width=4),
                dbc.Col([
                    html.Label("Text Prompt:", className="form-label fw-bold"),
                    dbc.Input(
                        id='cir-text-prompt',
                        placeholder="e.g., 'is wearing a red shirt', 'without the person'",
                        type="text",
                        className="mb-3"
                    ),
                    html.Label("Top N Results:", className="form-label fw-bold"),
                    dbc.Select(
                        id='cir-top-n',
                        options=[
                            {"label": "5 images", "value": 5},
                            {"label": "10 images", "value": 10},
                            {"label": "20 images", "value": 20}
                        ],
                        value=10,
                        className="mb-3"
                    )
                ], width=4),
                dbc.Col([
                    html.Label("Action:", className="form-label fw-bold"),
                    html.Br(),
                    dbc.Button(
                        [html.I(className="fas fa-search me-2"), "Start Retrieval"],
                        id='cir-search-button',
                        color="primary",
                        size="lg",
                        className="w-100",
                        disabled=True
                    ),
                    html.Div(id='cir-search-status', className="mt-2 status-indicator")
                ], width=4)
            ], className="mb-4 cir-interface"),
            # Query image preview
            html.Div(id='cir-query-preview', className="mb-3 query-preview"),
            # Results section
            html.Hr(),
            # Visualization control buttons (initially hidden)
            html.Div([
                dbc.Button("Visualize", id="cir-visualize-button", color="primary", className="me-2", n_clicks=0, disabled=False),
                dbc.Button("Hide", id="cir-hide-button", color="secondary", className="me-2", n_clicks=0, disabled=True)
            ], id="cir-vis-buttons", style={'display': 'none'}, className="mb-3"),
            html.Div(id='cir-results', children=[
                html.H5("Retrieved Images", className="mb-3"),
                html.Div("No results yet. Upload an image and enter a text prompt to start retrieval.", 
                        className="text-muted text-center p-4")
            ])
        ])
    ], className="mt-4")

    # Create app layout
    app.layout = html.Div(
        dbc.Container([
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
            ], className='h-100', justify='between'),
            # ], className='mt-4', justify='between'),
            # CIR Interface
            dbc.Row([
                dbc.Col(cir_interface, width=12)
            ], className='mt-4'),
            # Store for CIR search raw data
            dcc.Store(id='cir-search-data', data=None),
            # Store for selected image info in CIR mode
            dcc.Store(id='selected-image-data', data=None),
            # Store for selected gallery image IDs (for highlighting in images tab)
            dcc.Store(id='selected-gallery-image-ids', data=[])
        ], fluid=True, id='container'),
        style={'minHeight': '100vh', 'overflowY': 'auto', 'overflowX': 'hidden'}
    )

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