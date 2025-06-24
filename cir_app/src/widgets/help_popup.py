import dash_bootstrap_components as dbc
from dash import html

def create_help_popup():
    """Creates a help popup with information about the enhanced CIR application"""
    return dbc.Modal([
        dbc.ModalHeader([
            html.I(className="fas fa-question-circle text-primary me-2"),
            "Enhanced CIR Application Guide"
        ], style={'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', 'color': 'white'}),
        
        dbc.ModalBody([
            # Enhanced Introduction
            dbc.Alert([
                html.I(className="fas fa-magic text-info me-2"),
                html.Strong("Welcome to the Enhanced Composed Image Retrieval System! "),
                "This modernized interface features beautiful gradients, smooth animations, and intuitive interactions."
            ], color="light", className="border-start border-info border-4"),
            
            # Main Features Section
            html.Div([
                html.H5([
                    html.I(className="fas fa-star text-warning me-2"),
                    "Key Features"
                ], className="text-primary mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.I(className="fas fa-search fa-2x text-primary mb-2"),
                                html.H6("Smart Image Search", className="card-title"),
                                html.P("Upload an image and describe modifications to find similar images with enhanced visual feedback.", className="card-text small")
                            ])
                        ], className="h-100 border-0 shadow-sm")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.I(className="fas fa-brain fa-2x text-success mb-2"),
                                html.H6("AI Prompt Enhancement", className="card-title"),
                                html.P("Automatically improve your search queries with AI-powered suggestions and performance metrics.", className="card-text small")
                            ])
                        ], className="h-100 border-0 shadow-sm")
                    ], width=6)
                ], className="mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.I(className="fas fa-eye fa-2x text-info mb-2"),
                                html.H6("Advanced Visualization", className="card-title"),
                                html.P("Interactive scatter plots, histograms, and saliency maps with beautiful hover effects.", className="card-text small")
                            ])
                        ], className="h-100 border-0 shadow-sm")
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.I(className="fas fa-palette fa-2x text-warning mb-2"),
                                html.H6("Modern Design", className="card-title"),
                                html.P("Sleek gradients, smooth animations, and responsive design for an enhanced user experience.", className="card-text small")
                            ])
                        ], className="h-100 border-0 shadow-sm")
                    ], width=6)
                ])
            ], className="mb-4"),
            
            # How to Use Section
            html.Div([
                html.H5([
                    html.I(className="fas fa-play-circle text-success me-2"),
                    "How to Use"
                ], className="text-primary mb-3"),
                
                dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.I(className="fas fa-upload text-primary me-2"),
                        html.Strong("1. Upload Image: "),
                        "Drag & drop or click to upload your reference image. Watch for the smooth upload animation!"
                    ], className="border-0 bg-light"),
                    
                    dbc.ListGroupItem([
                        html.I(className="fas fa-edit text-info me-2"),
                        html.Strong("2. Write Description: "),
                        "Describe what modifications you want (e.g., 'wearing a red hat', 'in winter setting')"
                    ], className="border-0"),
                    
                    dbc.ListGroupItem([
                        html.I(className="fas fa-search text-success me-2"),
                        html.Strong("3. Search & Explore: "),
                        "Click search and enjoy the enhanced loading animations. Results appear with beautiful hover effects."
                    ], className="border-0 bg-light"),
                    
                    dbc.ListGroupItem([
                        html.I(className="fas fa-magic text-warning me-2"),
                        html.Strong("4. Enhance Prompts: "),
                        "Select result images and click 'Enhance prompt' for AI-powered improvements with performance metrics."
                    ], className="border-0"),
                    
                    dbc.ListGroupItem([
                        html.I(className="fas fa-chart-bar text-danger me-2"),
                        html.Strong("5. Visualize: "),
                        "Toggle visualization mode to see interactive plots with smooth transitions and hover details."
                    ], className="border-0 bg-light")
                ], flush=True)
            ], className="mb-4"),
            
            # Enhanced Interface Elements
            html.Div([
                html.H5([
                    html.I(className="fas fa-sparkles text-info me-2"),
                    "Enhanced Interface Elements"
                ], className="text-primary mb-3"),
                
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-mouse-pointer text-primary me-2"),
                            html.Strong("Hover Effects: "),
                            "Experience smooth hover animations on cards, buttons, and interactive elements."
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-palette text-success me-2"),
                            html.Strong("Gradient Design: "),
                            "Beautiful color gradients throughout the interface create visual depth and modern appeal."
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-expand-arrows-alt text-info me-2"),
                            html.Strong("Responsive Layout: "),
                            "Fullscreen modes available for detailed analysis with smooth transitions."
                        ])
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.I(className="fas fa-hand-pointer text-warning me-2"),
                            html.Strong("Interactive Feedback: "),
                            "Visual feedback for selections, loading states, and user interactions."
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-layer-group text-danger me-2"),
                            html.Strong("Depth & Shadows: "),
                            "Card elevation and shadow effects create visual hierarchy and modern aesthetics."
                        ], className="mb-2"),
                        html.Div([
                            html.I(className="fas fa-bolt text-primary me-2"),
                            html.Strong("Smooth Animations: "),
                            "Carefully crafted transitions and micro-interactions enhance usability."
                        ])
                    ], width=6)
                ])
            ], className="mb-4"),
            
            # Tips Section
            dbc.Alert([
                html.I(className="fas fa-lightbulb text-warning me-2"),
                html.Strong("Pro Tips: "),
                html.Ul([
                    html.Li("Use specific, descriptive language in your prompts for better results"),
                    html.Li("Try the AI prompt enhancement feature to improve search accuracy"),
                    html.Li("Explore different visualization modes to understand your data better"),
                    html.Li("Hover over elements to discover interactive features"),
                    html.Li("Use fullscreen modes for detailed analysis of visualizations")
                ], className="mb-0 mt-2")
            ], color="light", className="border-start border-warning border-4")
        ]),
        
        dbc.ModalFooter([
            dbc.Button([
                html.I(className="fas fa-rocket me-2"),
                "Start Exploring!"
            ], id="close-help", color="primary", className="shadow-sm")
        ])
    ], id="help-popup", size="lg", scrollable=True) 