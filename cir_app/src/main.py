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
import src.callbacks.saliency_callbacks
import src.callbacks.rank_delta
import src.callbacks.fullscreen

def run_ui():
    """Run the Dash UI application"""
    external_stylesheets = [
        dbc.themes.BOOTSTRAP,
        "https://use.fontawesome.com/releases/v5.15.4/css/all.css"
    ]
    app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
    
    # Create widgets
    help_popup_widget = help_popup.create_help_popup()
    projection_radio_buttons_widget = projection_radio_buttons.create_projection_radio_buttons()
    scatterplot_widget = scatterplot.create_scatterplot(config.DEFAULT_PROJECTION)
    wordcloud_widget = wordcloud.create_wordcloud()
    gallery_widget = gallery.create_gallery()
    histogram_widget = histogram.create_histogram()

    # -------------------------------------------------------------------------
    # RIGHT COLUMN TABS (Prompt Enhancement, Saliency, Token Attribution, Rank-Δ)
    # -------------------------------------------------------------------------
    right_tabs = dcc.Tabs(
        id='right-tabs',
        value='prompt-enhancement',
        children=[
            # Prompt Enhancement Tab
            dcc.Tab(label='Prompt Enhancement', value='prompt-enhancement', id='tab-prompt-enhancement', children=[
                html.Div([
                    html.Div(id='prompt-enhancement-content', style={
                        'overflowY': 'auto',
                        'flex': '1 1 auto',
                        'padding': '1rem',
                        'paddingBottom': '1rem'
                    }),
                    dcc.RadioItems(id='prompt-selection', options=[], value=None, style={'display': 'none'})
                ], style={
                    'display': 'flex',
                    'flexDirection': 'column',
                    'height': '100%'
                })
            ]),
            # Saliency Tab
            dcc.Tab(label='Saliency', value='saliency', id='tab-saliency', children=[
                html.Div([
                    html.Div(id='saliency-content', children=[
                        html.Div([
                            html.I(className="fas fa-brain text-info me-2"),
                            html.H5("Saliency Maps", className="d-inline"),
                            html.P("No saliency data available. Run a CIR query with SEARLE to generate saliency maps.",
                                   className="text-muted mt-2")
                        ], className="text-center p-4")
                    ], style={'flex': '1 1 auto', 'overflow': 'hidden'}),
                    html.Div(id='saliency-navigation', children=[
                        html.Div([
                            dbc.Button([html.I(className="fas fa-chevron-left me-1"), "Previous"],
                                       id='saliency-prev-btn', color='outline-primary', size='sm', disabled=True),
                            html.Span(id='saliency-current-info', className='mx-3 text-muted small fw-bold'),
                            dbc.Button(["Next ", html.I(className="fas fa-chevron-right ms-1")],
                                       id='saliency-next-btn', color='outline-primary', size='sm', disabled=True)
                        ], className='d-flex align-items-center justify-content-center gap-2 saliency-navigation-controls p-3')
                    ], style={'display': 'none'})
                ], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'})
            ]),
            # Token Attribution Tab (new)
            dcc.Tab(label='Token Attribution', value='token-attribution', id='tab-token-attribution', children=[
                html.Div(
                    id='token-attribution-content',
                    children=[
                        html.Div("Run a CIR query with saliency enabled to view token attributions.", className="text-muted p-4")
                    ],
                    style={'flex':'1 1 auto','overflowX': 'auto', 'overflowY': 'auto'}
                ),
                # Navigation handled in the right-column Token Attribution card
            ]),
            # Rank-Δ Tab
            dcc.Tab(label='Rank-Δ', value='rank-delta', id='tab-rank-delta', children=[
                html.Div(
                    id='rank-delta-content',
                    children=[
                        html.Div(
                            "Run prompt enhancement to view the rank-Δ matrix.",
                            className="text-muted p-4"
                        )
                    ],
                    style={'height': '100%', 'overflowX': 'auto', 'overflowY': 'auto'}
                )
            ])
        ]
    )

    # -------------------------------------------------------------------------
    # LEFT COLUMN (Inference, Top-K Images, Wordcloud / Histogram)
    # -------------------------------------------------------------------------
    left_word_hist_tabs = dcc.Tabs(
        id='left-tabs',
        value='histogram',
        children=[
            dcc.Tab(label='Histogram', value='histogram', id='left-tab-histogram', children=histogram_widget),
            dcc.Tab(label='Wordcloud', value='wordcloud', id='left-tab-wordcloud', children=wordcloud_widget)
        ]
    )

    # ------------------------- CIR CONTROLS CARD (upload & parameters) -------------------------
    cir_controls_card = dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.H6("Composed Image Retrieval", className="mb-0", style={'fontSize':'0.9rem'}),
                dbc.Button(
                    html.I(className="fas fa-expand fa-xs"),
                    id='cir-controls-expand-btn',
                    size='sm',
                    color='outline-secondary',
                    style={
                        'padding': '0.1rem',
                        'height': '1.0rem',
                        'width': '1.0rem',
                        'minWidth': 'auto',
                        'lineHeight': '1',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    },
                    class_name='ms-auto'
                ),
            ], className='d-flex align-items-center'),
        ),
        dbc.CardBody([
            html.Label("Upload Query Image:", className="form-label fw-bold small", style={'fontSize':'0.7rem'}),
            dcc.Upload(
                id='cir-upload-image',
                children=html.Div([
                    html.I(className="fas fa-cloud-upload-alt me-2"),
                    'Select'
                ], style={'fontSize':'0.7rem'}),
                style={
                    'width': '100%',
                    'height': '50px',
                    'lineHeight': '50px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '6px',
                    'textAlign': 'center',
                    'background': '#fafafa',
                    'cursor': 'pointer',
                    'fontSize': '0.75rem'
                },
                className='cir-upload-area mb-2',
                multiple=False,
                accept='image/*'
            ),
            html.Div(id='cir-upload-status', className="mb-2 small status-indicator"),

            html.Label("Text Prompt:", className="form-label fw-bold small", style={'fontSize':'0.7rem'}),
            dbc.Input(id='cir-text-prompt', placeholder="prompt", type="text", size='sm', className="mb-2", style={'fontSize':'0.7rem'}),

            html.Label("Top-N Results:", className="form-label fw-bold small"),
            dbc.Select(id='cir-top-n', options=[{"label": f"{n} images", "value": n} for n in (5,10,20)], value=10, size='sm', className="mb-2"),

            html.Label("Model:", className="form-label fw-bold small"),
            dbc.Select(id='custom-dropdown', options=[{"label": "SEARLE", "value": "SEARLE"}, {"label": "freedom", "value": "freedom"}], value="SEARLE", size='sm', className="mb-3"),

            dbc.Button("Start", id='cir-search-button', color="primary", size='sm', className="w-100 mb-2", disabled=True, style={'fontSize':'0.7rem'}),
            html.Div(id='cir-search-status', className="small status-indicator mb-2"),
            # Small preview thumbnail (filled by upload callback)
            html.Div(id='cir-upload-preview', className='mt-2')
        ], style={'height':'100%','overflow':'auto','flex':'1 1 auto'}),
    ], id='cir-controls-card', className="border-widget", style={'flex':'1 1 25%', 'overflow':'auto'})

    # ------------------------- CIR RESULTS CARD (preview + results) -------------------------
    cir_results_card = dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.H6("Query Results", className="mb-0", style={'fontSize': '0.9rem'}),
                dbc.Button(
                    html.I(className="fas fa-expand fa-xs"),
                    id='cir-results-expand-btn',
                    size='sm',
                    color='outline-secondary',
                    style={
                        'padding': '0.1rem',
                        'height': '1.0rem',
                        'width': '1.0rem',
                        'minWidth': 'auto',
                        'lineHeight': '1',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    },
                    class_name='ms-auto'
                ),
            ], className='d-flex align-items-center'),
        ),
        dbc.CardBody([
            html.Div(id='cir-query-preview', className="mb-2"),
            html.Div(id='cir-results', style={'flex':'1 1 auto', 'overflowY':'auto'}, children=[
                html.Small("No results yet. Upload an image and enter a text prompt to start retrieval.", className="text-muted")
            ]),
            html.Hr(className="my-2"),
            dbc.Button("Enhance prompt", id='enhance-prompt-button', color='secondary', size='sm', disabled=True, className="w-100 mb-1", style={'fontSize':'0.7rem'})
        ], style={'height':'100%','overflow':'auto','flex':'1 1 auto'}),
    ], id='cir-results-card', className="border-widget mt-2", style={'flex':'1 1 25%', 'overflow':'auto'})

    # Wordcloud/Histogram card with fullscreen capability
    left_wh_card = dbc.Card([
        dbc.CardHeader(
            html.Div([
                html.H6("Histogram / Wordcloud", className="mb-0", style={'fontSize': '0.9rem'}),
                dbc.Button(
                    html.I(className="fas fa-expand fa-xs"),
                    id='wh-expand-btn',
                    size='sm',
                    color='outline-secondary',
                    style={
                        'padding': '0.1rem',
                        'height': '1.0rem',
                        'width': '1.0rem',
                        'minWidth': 'auto',
                        'lineHeight': '1',
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'center'
                    },
                    class_name='ms-auto'
                ),
            ], className='d-flex align-items-center'),
        ),
        dbc.CardBody(left_word_hist_tabs, style={'height':'100%','overflow':'auto','flex':'1 1 auto'}),
    ], id='hist-wh-card', className="border-widget mt-2", style={'flex':'1 1 25%', 'overflow':'auto'})

    left_column = html.Div(
        id='cir-interface',  # Anchor target for "Run CIR" button
        children=[
            cir_controls_card,
            cir_results_card,
            left_wh_card,
            html.Div(id='gallery', style={'display':'none'})  # Hidden placeholder to satisfy callbacks
        ],
        className='d-flex flex-column h-100'
    )

    # ------------------------------- APP LAYOUT -------------------------------
    app.layout = html.Div(
        dbc.Container([
            help_popup_widget,
            html.Div(id='model-change-flag', style={'display': 'none'}),
            dbc.Stack([
                projection_radio_buttons_widget,
                html.Div([
                    # Hidden placeholder for legacy callbacks – kept for ID consistency
                    html.A(
                        dbc.Button('Run CIR',
                                   id='cir-run-button',
                                   color='info',
                                   class_name='header-button',
                                   style={'display': 'none'}),
                        href='#cir-interface',
                        style={'textDecoration': 'none', 'display': 'none'}
                    ),
                    # Visualize button is always visible but starts disabled until results are available
                    dbc.Button('Visualize CIR results', 
                               id='cir-toggle-button',
                               color="success",
                               class_name="header-button",
                               disabled=True,
                               style={'display': 'block', 'color': 'black'}),
                    dbc.Button('Deselect everything', 
                               id='deselect-button', 
                               class_name="btn btn-outline-primary header-button"),
                    dbc.Button('Help', 
                               id='help-button', 
                               class_name="btn btn-outline-primary header-button")
                ], className='ms-auto d-flex gap-2 align-items-center'),
            ], id='header', direction="horizontal"),
            dbc.Row([
                dbc.Col(left_column, width=3, className='main-col'),
                dbc.Col(scatterplot_widget, width=6, className='main-col'),
                dbc.Col(
                    # RIGHT COLUMN STACKED COMPONENTS (no tabs)
                    html.Div([
                        # Prompt Enhancement card with fullscreen capability
                        dbc.Card([
                            dbc.CardHeader(
                                html.Div([
                                    html.H6("Prompt Enhancement", className="mb-0", style={'fontSize': '0.85rem'}),
                                    dbc.Button(
                                        html.I(className="fas fa-expand fa-xs"),
                                        id='prompt-enh-expand-btn',
                                        size='sm',
                                        color='outline-secondary',
                                        style={
                                            'padding': '0.1rem',
                                            'height': '1.0rem',
                                            'width': '1.0rem',
                                            'minWidth': 'auto',
                                            'lineHeight': '1',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center'
                                        },
                                        class_name='ms-auto'
                                    ),
                                ], className='d-flex align-items-center'),
                            ),
                            dbc.CardBody([
                                html.Div(id='prompt-enhancement-content', style={'overflow': 'auto', 'flex': '1 1 auto', 'minHeight': 0}),
                                dcc.RadioItems(id='prompt-selection', options=[], value=None, style={'display':'none'})
                            ], style={'display': 'flex', 'flexDirection': 'column', 'padding': '0.5rem'}),
                        ], id='prompt-enh-card', className='border-widget', style={'flex': '1 1 25%', 'display': 'flex', 'flexDirection': 'column', 'height': '25%', 'maxHeight': '25%', 'overflow': 'auto'}),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Div([
                                    html.H6("Saliency", className="mb-0", style={'fontSize': '0.85rem'}),
                                    html.Div([
                                        dbc.Button([html.I(className="fas fa-chevron-left me-1"), "Prev"],
                                                id='saliency-prev-btn', color='outline-primary', size='sm', disabled=True, style={'fontSize': '0.6rem', 'padding': '0.2rem 0.4rem'}),
                                        html.Span(id='saliency-current-info', className='mx-2 text-muted small fw-bold', style={'fontSize': '0.6rem'}),
                                        dbc.Button(["Next ", html.I(className="fas fa-chevron-right ms-1")],
                                                id='saliency-next-btn', color='outline-primary', size='sm', disabled=True, style={'fontSize': '0.6rem', 'padding': '0.2rem 0.4rem'})
                                    ], className="d-flex align-items-center ms-3", id='saliency-navigation-controls'),

                                    dbc.Button(
                                        html.I(className="fas fa-expand fa-xs"),
                                        id='saliency-expand-btn',
                                        size='sm',
                                        color='outline-secondary',
                                        className='ms-auto',
                                        style={
                                            'padding': '0.1rem',
                                            'height': '1.0rem',
                                            'width': '1.0rem',
                                            'minWidth': 'auto',
                                            'lineHeight': '1',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center'
                                        },
                                    )
                                ], className='d-flex align-items-center')
                            ),
                            dbc.CardBody([
                                html.Div(id='saliency-content', style={'overflow': 'auto', 'flex':'1 1 auto', 'minHeight': '0'}),
                            ], style={'display': 'flex', 'flexDirection': 'column', 'padding': '0.5rem'}),
                        ], id='saliency-card', className='border-widget mt-2', style={'flex':'1 1 25%', 'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'maxHeight': '25%', 'overflow': 'auto'}),

                        # Token Attribution card with fullscreen capability
                        dbc.Card([
                            dbc.CardHeader(
                                html.Div([
                                    html.H6("Token Attribution", className="mb-0", style={'fontSize': '0.85rem'}),

                                    html.Div([
                                        dbc.Button([html.I(className="fas fa-chevron-left me-1"), "Prev"],
                                                id='ta-prev-btn', color='outline-primary', size='sm', disabled=True, style={'fontSize': '0.6rem', 'padding': '0.2rem 0.4rem'}),
                                        html.Span(id='token-attribution-current-info', className='mx-2 text-muted small fw-bold', style={'fontSize': '0.6rem'}),
                                        dbc.Button(["Next ", html.I(className="fas fa-chevron-right ms-1")],
                                                id='ta-next-btn', color='outline-primary', size='sm', disabled=True, style={'fontSize': '0.6rem', 'padding': '0.2rem 0.4rem'})
                                    ], className="d-flex align-items-center ms-3", id='token-attribution-controls'),

                                    dbc.Button(
                                        html.I(className="fas fa-expand fa-xs"),
                                        id='token-attr-expand-btn',
                                        size='sm',
                                        color='outline-secondary',
                                        className='ms-auto',
                                        style={
                                            'padding': '0.1rem',
                                            'height': '1.0rem',
                                            'width': '1.0rem',
                                            'minWidth': 'auto',
                                            'lineHeight': '1',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center'
                                        },
                                    )
                                ], className='d-flex align-items-center')
                            ),
                            dbc.CardBody([
                                html.Div(id='token-attribution-content', style={'overflow': 'auto', 'flex': '1 1 auto', 'minHeight': 0})
                            ], style={'display': 'flex', 'flexDirection': 'column', 'padding': '0.5rem'}),
                        ], id='token-attr-card', className='border-widget mt-2', style={'flex':'1 1 25%', 'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'maxHeight': '25%', 'overflow': 'auto'}),

                        dbc.Card([
                            dbc.CardHeader(
                                html.Div([
                                    html.H6("Rank-Δ", className="mb-0", style={'fontSize': '0.85rem'}),
                                    dbc.Button(
                                        html.I(className="fas fa-expand fa-xs"),
                                        id='rank-delta-expand-btn',
                                        size='sm',
                                        color='outline-secondary',
                                        style={
                                            'padding': '0.1rem',
                                            'height': '1.0rem',
                                            'width': '1.0rem',
                                            'minWidth': 'auto',
                                            'lineHeight': '1',
                                            'display': 'flex',
                                            'alignItems': 'center',
                                            'justifyContent': 'center'
                                        },
                                        class_name='ms-auto'
                                    ),
                                ], className='d-flex align-items-center'),
                            ),
                            dbc.CardBody([
                                html.Div(id='rank-delta-content', style={'overflow': 'auto', 'flex': '1 1 auto', 'minHeight': 0})
                            ], style={'display': 'flex', 'flexDirection': 'column', 'padding': '0.5rem'}),
                        ], id='rank-delta-card', className='border-widget mt-2', style={'flex': '1 1 25%', 'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'maxHeight': '25%', 'overflow': 'auto'})
                    ], className='d-flex flex-column h-100'),
                    width=3, className='main-col'
                )
            ], className='h-100', justify='between'),
            
            # Enhanced Prompt Analysis - Separate section below everything
            html.Div([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="fas fa-magic me-2"),
                            "Enhanced Prompt Analysis"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.P("Enhanced prompts and analysis will appear here after clicking the enhance button above.", 
                               className="text-muted mb-3"),
                        html.Div(id='cir-enhance-results', children=[], className='mt-3')
                    ])
                ], className="border-widget")
            ], className='mt-4 mb-4'),
            
            # Store for CIR search raw data
            dcc.Store(id='cir-search-data', data=None),
            # Store for CIR toggle state (True = visualized, False = hidden)
            dcc.Store(id='cir-toggle-state', data=False),
            # Store for selected image info in CIR mode
            dcc.Store(id='selected-image-data', data=None),
            # Store for selected gallery image IDs (for highlighting in images tab)
            dcc.Store(id='selected-gallery-image-ids', data=[]),
            # Store for the selected CIR result image IDs (supports multiple selections) to support prompt enhancement
            dcc.Store(id='cir-selected-image-ids', data=[]),
            # Store for enhanced prompts data and results
            dcc.Store(id='cir-enhanced-prompts-data', data=None),
            # Store for visualization toggle state
            dcc.Store(id='viz-mode', data=False),
            # Store for ids of images selected in visualization mode
            dcc.Store(id='viz-selected-ids', data=[]),
            # Store for saliency data and current candidate index
            dcc.Store(id='saliency-data', data=None),
            dcc.Store(id='saliency-current-index', data=0),
            # Store for class selected in histogram (None when nothing is selected)
            dcc.Store(id='selected-histogram-class', data=None),
            # Store for Query Results fullscreen state
            dcc.Store(id='cir-results-fullscreen', data=False),
            # Store for Histogram/Wordcloud fullscreen state
            dcc.Store(id='hist-wh-fullscreen', data=False),
            # Store for Prompt Enhancement fullscreen state
            dcc.Store(id='prompt-enh-fullscreen', data=False),
            # Store for Rank-Delta fullscreen state
            dcc.Store(id='rank-delta-fullscreen', data=False),
            # Store for Saliency fullscreen state
            dcc.Store(id='saliency-fullscreen', data=False),
            # Store for Token Attribution fullscreen state
            dcc.Store(id='token-attr-fullscreen', data=False),
            # Store for CIR Controls fullscreen state
            dcc.Store(id='cir-controls-fullscreen', data=False),
            # Store for card ID
            dcc.Store(id='card-id', data=None),
            # Store for token attribution index in the list of dcc.Store components (search for saliency-data store area)
            dcc.Store(id='token-attribution-index', data=0),
            # Before fullscreen styles
            dcc.Store(id='cir-results-card-style'),
            dcc.Store(id='hist-wh-card-style'),
            dcc.Store(id='prompt-enh-card-style'),
            dcc.Store(id='rank-delta-card-style'),
            dcc.Store(id='token-attr-card-style'),
            dcc.Store(id='cir-controls-card-style'),
            dcc.Store(id='saliency-card-style')
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
        Dataset.load(reload=True)

    print('Starting Dash application')
    run_ui()

if __name__ == '__main__':
    main() 