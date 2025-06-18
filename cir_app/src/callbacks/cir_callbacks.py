import base64
import threading
import tempfile
from io import BytesIO
from dash import callback, Input, Output, State, html, callback_context
import dash_bootstrap_components as dbc
from PIL import Image
from src import config
import os
import sys
from src.Dataset import Dataset
import torch
import torch.nn.functional as F
import pickle
import clip
from dash.exceptions import PreventUpdate
from dash import ALL
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from dash import dcc
import plotly.graph_objects as go
from src.widgets import gallery, wordcloud, histogram, scatterplot

@callback(
    [Output('cir-upload-status', 'children'),
     Output('cir-query-preview', 'children'),
     Output('cir-search-button', 'disabled')],
    [Input('cir-upload-image', 'contents'),
     Input('cir-text-prompt', 'value')],
    [State('cir-upload-image', 'filename')]
)
def update_upload_status(upload_contents, text_prompt, filename):
    """Handle image upload and update UI components"""
    if upload_contents is None:
        return (html.Div("No image uploaded", className="text-muted small"), html.Div(), True)
    try:
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))
        preview = html.Div([
            html.H6("Query Image Preview:", className="mb-2"),
            dbc.Card(dbc.CardBody([
                html.Img(src=upload_contents, className="img-fluid", style={'max-width':'200px','max-height':'200px'}),
                html.P(f"Filename: {filename}", className="small text-muted mt-2 mb-0"),
                html.P(f"Size: {img.size[0]} x {img.size[1]}", className="small text-muted mb-0")
            ]), style={'width': 'fit-content'})
        ])
        uploaded = html.Div([html.I(className="fas fa-check-circle text-success me-2"), "Image uploaded successfully"], className="text-success small")
        disabled = not (upload_contents and text_prompt and text_prompt.strip())
        return uploaded, preview, disabled
    except Exception as e:
        error = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Error processing image: {e}"], className="text-danger small")
        return error, html.Div(), True

@callback(
    Output('cir-search-button', 'disabled', allow_duplicate=True),
    [Input('cir-text-prompt', 'value')],
    [State('cir-upload-image', 'contents')],
    prevent_initial_call=True
)
def update_search_button_state(text_prompt, upload_contents):
    """Enable/disable search button based on input validation"""
    if upload_contents and text_prompt and text_prompt.strip():
        return False
    return True

@callback(
    [Output('cir-results', 'children'),
     Output('cir-search-status', 'children'),
     Output('cir-search-data', 'data'),
     Output('cir-toggle-button', 'style', allow_duplicate=True),
     Output('cir-run-button', 'style'),
     Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data', allow_duplicate=True),
     Output('saliency-data', 'data')],
    [Input('cir-search-button', 'n_clicks')],
    [State('cir-upload-image', 'contents'),
     State('cir-text-prompt', 'value'),
     State('cir-top-n', 'value'),
     State('custom-dropdown', 'value')],
    prevent_initial_call=True
)
def perform_cir_search(n_clicks, upload_contents, text_prompt, top_n, selected_model):
    """Perform CIR search using the SEARLE ComposedImageRetrievalSystem"""
    if not upload_contents or not text_prompt:
        empty = html.Div("No results yet. Upload an image and enter a text prompt to start retrieval.", className="text-muted text-center p-4")
        # Show Run CIR button, hide visualize button, clear enhance results and data
        return empty, html.Div(), None, {'display': 'none', 'color': 'black'}, {'display': 'block', 'color': 'black'}, [], None, None
    try:
        print(f"Starting CIR search with model: {selected_model}, prompt: '{text_prompt}', top_n: {top_n}")
        
        # Decode and save query image
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(decoded)
        tmp.close()
        print(f"Query image saved to: {tmp.name}")

        print(f"Selected CIR model: {selected_model}")

        # Use saliency-enabled CIR query
        from src.saliency import perform_cir_with_saliency, get_saliency_status_message
        print("Calling perform_cir_with_saliency...")
        results, saliency_data = perform_cir_with_saliency(
            temp_image_path=tmp.name,
            text_prompt=text_prompt,
            top_n=top_n,
            selected_model=selected_model
        )
        print(f"CIR search completed. Got {len(results)} results")

        # Build result cards using paths from the loaded dataset
        cards = []
        df = Dataset.get()
        print(f"Dataset has {len(df)} entries")
        
        for card_index, (img_name, score) in enumerate(results):
            img_path = None
            try:
                # Try numeric index lookup
                idx = int(img_name)
                if idx in df.index:
                    img_path = df.loc[idx]['image_path']
            except Exception:
                # Try string index lookup
                if img_name in df.index:
                    img_path = df.loc[img_name]['image_path']
            
            # Skip if path not found or doesn't exist
            if not img_path or not os.path.exists(img_path):
                print(f"Warning: Could not find image path for {img_name}")
                continue
                
            # Load and encode image
            try:
                data = open(img_path, 'rb').read()
                src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
            
            # Create card body with score (like original CIR)
            card_body = dbc.CardBody([
                html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
                html.P(f"Score: {score:.4f}", className='small text-center mt-1')
            ])
            
            # First image (top-1) is not clickable
            if card_index == 0:
                inner_card = dbc.Card(card_body, className='result-card')
                card = html.Div(
                    [inner_card,
                     html.Div("TOP-1", className='badge bg-warning position-absolute top-0 start-0 m-1')],
                    className='result-card-wrapper position-relative',
                    style={'cursor': 'default'}  # Override pointer cursor for first card
                )
            else:
                # Other images are clickable
                inner_card = dbc.Card(card_body, className='result-card')
                card = html.Div(
                    inner_card,
                    id={'type': 'cir-result-card', 'index': img_name},
                    n_clicks=0,
                    className='result-card-wrapper'
                )
            cards.append(card)

        print(f"Created {len(cards)} result cards")

        rows = []
        for i in range(0, len(cards), 5):
            chunk = cards[i:i+5]
            cols = [dbc.Col(c, width=12//len(chunk)) for c in chunk]
            rows.append(dbc.Row(cols, className='mb-3'))

        results_div = html.Div([html.H5("Retrieved Images", className="mb-3")] + rows)
        
        # Create status message with saliency information
        status_messages = [html.I(className="fas fa-check-circle text-success me-2"), f"Retrieved {len(cards)} images"]
        saliency_status = get_saliency_status_message(saliency_data)
        if saliency_status:
            status_messages.extend([html.Br(), html.I(className="fas fa-brain text-info me-2"), saliency_status])
        
        status = html.Div(status_messages, className="text-success small")
        
        # Reduce saliency data to a lightweight summary (only save_directory) to avoid large JSON payloads
        saliency_summary = None
        if saliency_data and isinstance(saliency_data, dict):
            save_dir = saliency_data.get('save_directory')
            if save_dir:
                saliency_summary = {'save_directory': save_dir}
        
        # Prepare store data for visualization - simplified version
        # Map retrieved image names to DataFrame indices
        topk_ids = []
        for img_name, _ in results:
            try:
                idx = int(img_name)
                if idx in df.index:
                    topk_ids.append(idx)
            except:
                if img_name in df.index:
                    topk_ids.append(img_name)
        top1_id = topk_ids[0] if topk_ids else None
        
        # Delete temporary query image file
        os.unlink(tmp.name)
        print("Temporary query image deleted")
        
        # Simplified store data - we'll compute embeddings later if needed for visualization
        store_data = {
            'topk_ids': topk_ids,
            'top1_id': top1_id,
            'umap_x_query': None,  # Will be computed when visualization is enabled
            'umap_y_query': None,
            'umap_x_final_query': None,
            'umap_y_final_query': None,
            'tsne_x_query': None,
            'tsne_y_query': None,
            'text_prompt': text_prompt,
            'top_n': top_n
        }
        
        print("CIR search callback completed successfully")
        # Show visualize button, hide Run CIR, clear enhance data
        return results_div, status, store_data, {'display': 'block', 'color': 'black'}, {'display': 'none', 'color': 'black'}, [], None, saliency_summary
    except Exception as e:
        print(f"CIR search error: {e}")
        import traceback
        traceback.print_exc()
        err = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Retrieval error: {e}"], className="text-danger small")
        # On error, hide visualize button, show Run CIR, clear enhance results and data
        return html.Div("Error occurred during image retrieval.", className="text-danger text-center p-4"), err, None, {'display': 'none', 'color': 'black'}, {'display': 'block', 'color': 'black'}, [], None, None

# Button toggle callback for CIR visualization
@callback(
    [Output('cir-toggle-button', 'children'),
     Output('cir-toggle-button', 'color'),
     Output('cir-toggle-button', 'style', allow_duplicate=True),
     Output('cir-toggle-state', 'data')],
    Input('cir-toggle-button', 'n_clicks'),
    State('cir-toggle-state', 'data'),
    prevent_initial_call=True
)
def toggle_cir_visualization(n_clicks, current_state):
    """Toggle between visualize and hide CIR results"""
    if n_clicks is None:
        return 'Visualize CIR results', 'success', {'display': 'block', 'color': 'black'}, False
    
    # Toggle state
    new_state = not current_state
    
    if new_state:  # Now visualizing
        return 'Hide CIR results', 'warning', {'display': 'block', 'color': 'black'}, True
    else:  # Now hidden
        return 'Visualize CIR results', 'success', {'display': 'block', 'color': 'black'}, False

# Disabled dynamic enhance button creation; static button is in layout

# New callback to update the 'Enhance prompt' button based on selection
@callback(
    [Output('enhance-prompt-button', 'disabled'),
     Output('enhance-prompt-button', 'color')],
    Input({'type': 'cir-result-card', 'index': ALL}, 'className')
)
def update_enhance_button_state(wrapper_classnames):
    """
    Enable the Enhance prompt button when a result is selected; otherwise keep it disabled.
    """
    # If any wrapper has the 'selected' class, enable button
    if any('selected' in cn for cn in wrapper_classnames):
        return False, 'primary'
    return True, 'secondary'

@callback(
    [Output({'type': 'cir-result-card', 'index': ALL}, 'className'),
     Output('cir-selected-image-id', 'data')],
    Input({'type': 'cir-result-card', 'index': ALL}, 'n_clicks'),
    [State({'type': 'cir-result-card', 'index': ALL}, 'className')],
    prevent_initial_call=True
)
def toggle_cir_result_selection(n_clicks_list, current_classnames):
    """
    Toggle selection highlight for CIR result cards, allowing only one selected at a time.
    Clicking the same card again will deselect it.
    """
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    # Get index of clicked card
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        selected_dict = json.loads(triggered_id)
        selected_index = selected_dict.get('index')
    except Exception:
        selected_index = None
    
    # Check if the clicked card is already selected
    clicked_card_currently_selected = False
    for i, input_dict in enumerate(ctx.inputs_list[0]):
        if input_dict['id'].get('index') == selected_index:
            if 'selected' in current_classnames[i]:
                clicked_card_currently_selected = True
            break
    
    # Build className list in order of inputs
    class_names = []
    for input in ctx.inputs_list[0]:
        idx = input['id'].get('index')
        if idx == selected_index:
            # If already selected, deselect it; otherwise select it
            if clicked_card_currently_selected:
                class_names.append('result-card-wrapper')  # Deselect
            else:
                class_names.append('result-card-wrapper selected')  # Select
        else:
            class_names.append('result-card-wrapper')  # Deselect all others
    
    # Determine selected image id or deselect
    if clicked_card_currently_selected:
        selected_image_id = None
    else:
        selected_image_id = selected_index
    return class_names, selected_image_id

# New callback to enhance the user prompt and evaluate against the selected image
@callback(
    [Output('cir-search-status', 'children', allow_duplicate=True),
     Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data')],
    Input('enhance-prompt-button', 'n_clicks'),
    [State('cir-search-data', 'data'), State('cir-selected-image-id', 'data')],
    prevent_initial_call=True
)
def enhance_prompt(n_clicks, search_data, selected_image_id):
    """
    Enhance the user's prompt via a small LLM, compare each to the selected image, choose the best,
    rerun CIR with that prompt, and display diagnostics.
    """
    import os
    # Guard against missing data
    if not search_data or selected_image_id is None:
        raise PreventUpdate

    # Reconstruct query image file
    _, content_string = search_data['upload_contents'].split(',')
    decoded = base64.b64decode(content_string)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    tmp.write(decoded)
    tmp.close()

    # Prepare LLM for prompt enhancement using Mistral-7B-Instruct
    N = config.ENHANCEMENT_CANDIDATE_PROMPTS  # Number of candidate prompts to generate
    original_prompt = search_data['text_prompt']
    MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_dtype = torch.float16 if DEVICE == 'cuda' else torch.float32
    print(f"Loading enhancement model {MODEL_NAME} on {DEVICE}...")  # Debug log
    os.environ['HF_TOKEN'] = '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'
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        device_map='auto'
    )
    instruction = f"""
    You are an assistant helping improve short prompts for image retrieval. 
    Given a query like: "{original_prompt}", generate one short, reworded version 
    that retains the original meaning but adds slight variety or detail. 
    Do NOT describe scenes or characters. Just rephrase the original style-focused prompt.

    Original prompt: "{original_prompt}"

    Return only one short enhanced prompt enclosed inside <ANSWER> </ANSWER> tags.
    """

    formatted_prompt = f"<s>[INST] {instruction.strip()} [/INST]"
    inputs = tokenizer(formatted_prompt, return_tensors='pt').to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=True,
        temperature=1.2,
        top_p=0.8,
        top_k=50,
        repetition_penalty=1.1,
        num_return_sequences=N,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    print(f"Enhancement results: {len(results)}")

    # Extract enhanced prompts from between <ANSWER> tags (case-insensitive)
    prompts = []
    for i, result in enumerate(results):
        result = result.lower()
        # First extract only the response part after [/INST]
        inst_idx = result.find('[/inst]')
        if inst_idx == -1:
            print(f"No [/INST] found in result {i+1}")
            continue
        
        response_part = result[inst_idx + 7:]  # Skip '[/INST]'
        
        # Now look for ANSWER tags in the response part only
        start_tag = '<answer>'
        end_tag = '</answer>'
        start_idx = response_part.find(start_tag)
        end_idx = response_part.find(end_tag)
        
        if start_idx != -1 and end_idx != -1:
            prompt = response_part[start_idx + len(start_tag):end_idx].strip()
            if prompt:  # Only add non-empty prompts
                prompts.append(prompt)
            else:
                print(f"Prompt was empty after stripping for result {i+1}")
    
    # If no valid prompts found, use original as fallback
    if not prompts:
        print("No prompts extracted, using original as fallback")
        prompts = [original_prompt]
    
    # clean up prompts
    prompts = [p.strip() for p in prompts]
    prompts = [p for p in prompts if p]
    # remove '"' from prompts
    prompts = [p.replace('"', '') for p in prompts]
    # Remove duplicates
    prompts = list(set(prompts))
    print(f"Final prompts list: {prompts}")

    # Score each candidate prompt and store full results
    sims = []
    ranks = []  # List to store the rank (position) of the selected image
    all_prompt_results = []
    
    # Use saliency-enabled enhanced prompt processing
    from src.saliency import perform_enhanced_prompt_cir_with_saliency
    all_prompt_results, enhanced_saliency_data = perform_enhanced_prompt_cir_with_saliency(
        temp_image_path=tmp.name,
        enhanced_prompts=prompts,
        top_n=search_data['top_n'],
        selected_image_id=selected_image_id
    )
    
    # Process results for scoring
    for i, (p, full_prompt_results) in enumerate(zip(prompts, all_prompt_results)):
        # Find the similarity score for the ideal image in these results
        ideal_score = None
        position = None
        for idx, (name, score) in enumerate(full_prompt_results):
            if str(name) == str(selected_image_id):
                ideal_score = score
                position = idx + 1
                break
        if ideal_score is None:
            ideal_score = 0.0
        if position is None:
            position = len(full_prompt_results) + 1  # beyond top-N
        sims.append(ideal_score)
        ranks.append(position)

    # Select best prompt by lowest rank (best position)
    best_idx = min(range(len(prompts)), key=lambda i: ranks[i])
    best_prompt = prompts[best_idx]
    best_sim_score = sims[best_idx]
    best_position = ranks[best_idx]
    print(f"Selected best prompt: '{best_prompt}' with score: {best_sim_score}")

    # Get results for best prompt (already computed)
    full_results = all_prompt_results[best_idx]
    
    # Clean up temporary file
    os.unlink(tmp.name)

    # Status message with icon
    status_messages = [
        html.I(className="fas fa-magic text-success me-2"),
        "Enhanced prompt generated successfully! See analysis below."
    ]
    
    # Add saliency status if available
    if enhanced_saliency_data:
        from src.saliency import get_saliency_status_message
        enhanced_saliency_status = get_saliency_status_message(enhanced_saliency_data)
        if enhanced_saliency_status:
            status_messages.extend([html.Br(), html.I(className="fas fa-brain text-info me-2"), enhanced_saliency_status])
    
    status = html.Div(status_messages, className="text-success small mb-3")

    # Create enhanced table rows with highlighting for best prompt and action buttons
    table_rows = []
    for i, (p, s) in enumerate(zip(prompts, sims)):
        view_button = dbc.Button(
            [html.I(className="fas fa-eye me-1"), "View"],
            id={'type': 'enhanced-prompt-view', 'index': i},
            size="sm",
            color="outline-primary" if i != best_idx else "primary",
            className="btn-sm"
        )
        
        if i == best_idx:  # Highlight best prompt
            row = html.Tr([
                html.Td([html.I(className="fas fa-crown text-warning me-2"), p], className="fw-bold"),
                html.Td([html.Span(str(ranks[i]), className="badge bg-success")]),
                html.Td([html.Span(f"{s:.4f}", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(str(ranks[i]), className="badge bg-secondary")),
                html.Td(html.Span(f"{s:.4f}", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with better styling and action column
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
            html.Th([html.I(className="fas fa-hashtag me-2"), "Position"], className="bg-light"),
            html.Th([html.I(className="fas fa-chart-line me-2"), "Similarity Score"], className="bg-light"),
            html.Th([html.I(className="fas fa-cogs me-2"), "Actions"], className="bg-light")
        ]), className="thead-light"),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, responsive=True, className="mb-4 shadow-sm")

    # Build image cards for CIR results of best prompt
    df = Dataset.get()
    cards = []
    for img_name, score in full_results:
        img_path = None
        try:
            idx = int(img_name)
            if idx in df.index:
                img_path = df.loc[idx]['image_path']
        except:
            if img_name in df.index:
                img_path = df.loc[img_name]['image_path']
        if not img_path or not os.path.exists(img_path):
            continue
        data = open(img_path, "rb").read()
        src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        # Create card body with score (like original CIR)
        card_body = dbc.CardBody([
            html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
            html.P(f"Score: {score:.4f}", className='small text-center mt-1')
        ])
        card = dbc.Card(card_body, className='result-card', style={'cursor': 'default'})
        cards.append(dbc.Col(card, width=2))

    # Arrange cards into rows of up to 6
    rows = []
    for i in range(0, len(cards), 6):
        rows.append(dbc.Row(cards[i:i+6], className="g-2 mb-3"))
    images_grid = html.Div(rows)

    # Build enhance results children with improved styling
    enhance_children = [
        # Main title with icon
        html.Div([
            html.I(className="fas fa-brain text-primary me-2"),
            html.H5("Enhanced Prompt Analysis", className="d-inline text-primary")
        ], className="mt-3 mb-4"),
        
        # Candidates section
        html.Div([
            html.H6([html.I(className="fas fa-list-alt me-2"), "Candidate Prompts & Scores"], className="mb-3 text-secondary"),
            candidates_table
        ]),
        
        # Best prompt highlight card
        dbc.Alert([
            html.H6([html.I(className="fas fa-trophy text-warning me-2"), "Selected Best Prompt"], className="alert-heading mb-2"),
            html.P(f'"{best_prompt}"', className="mb-1 font-monospace"),
            html.Small(f"Position: {best_position} | Similarity Score: {best_sim_score:.4f}", className="text-muted")
        ], color="light", className="border-start border-warning border-4"),
        
        # Results section
        html.Div([
            html.H6([html.I(className="fas fa-images me-2"), "CIR Results for Best Prompt"], className="mt-4 mb-3 text-secondary"),
            images_grid
        ])
    ]

    # Store data for enhanced prompts
    enhanced_prompts_data = {
        'prompts': prompts,
        'similarities': sims,
        'positions': ranks,
        'all_results': all_prompt_results,
        'best_idx': best_idx,
        'currently_viewing': best_idx  # Default to showing best prompt results
    }
    
    return status, enhance_children, enhanced_prompts_data

# New callback to handle enhanced prompt view button clicks
@callback(
    [Output('cir-enhance-results', 'children', allow_duplicate=True),
     Output('cir-enhanced-prompts-data', 'data', allow_duplicate=True)],
    [Input({'type': 'enhanced-prompt-view', 'index': ALL}, 'n_clicks')],
    [State('cir-enhanced-prompts-data', 'data')],
    prevent_initial_call=True
)
def update_enhanced_prompt_view(n_clicks_list, enhanced_data):
    """Update the enhanced prompt results display when a view button is clicked"""
    if not enhanced_data or not any(n_clicks_list):
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the clicked button index
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        clicked_dict = json.loads(triggered_id)
        clicked_index = clicked_dict.get('index')
    except Exception:
        raise PreventUpdate
    
    # Update the currently viewing index
    enhanced_data['currently_viewing'] = clicked_index
    
    # Get data for the clicked prompt
    prompts = enhanced_data['prompts']
    similarities = enhanced_data['similarities']
    positions = enhanced_data['positions']
    all_results = enhanced_data['all_results']
    best_idx = enhanced_data['best_idx']
    
    clicked_prompt = prompts[clicked_index]
    clicked_similarity = similarities[clicked_index]
    clicked_position = positions[clicked_index]
    clicked_results = all_results[clicked_index]
    
    # Build image cards for the clicked prompt results
    df = Dataset.get()
    cards = []
    for img_name, score in clicked_results:
        img_path = None
        try:
            idx = int(img_name)
            if idx in df.index:
                img_path = df.loc[idx]['image_path']
        except:
            if img_name in df.index:
                img_path = df.loc[img_name]['image_path']
        if not img_path or not os.path.exists(img_path):
            continue
        data = open(img_path, "rb").read()
        src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
        # Create card body with score (like original CIR)
        card_body = dbc.CardBody([
            html.Img(src=src, className='img-fluid', style={'maxHeight':'150px','width':'auto'}),
            html.P(f"Score: {score:.4f}", className='small text-center mt-1')
        ])
        card = dbc.Card(card_body, className='result-card', style={'cursor': 'default'})
        cards.append(dbc.Col(card, width=2))

    # Arrange cards into rows of up to 6
    rows = []
    for i in range(0, len(cards), 6):
        rows.append(dbc.Row(cards[i:i+6], className="g-2 mb-3"))
    images_grid = html.Div(rows)

    # Create enhanced table rows with updated button states
    table_rows = []
    for i, (p, s) in enumerate(zip(prompts, similarities)):
        # Highlight the currently viewing button
        if i == clicked_index:
            button_color = "primary"
            button_text = [html.I(className="fas fa-eye me-1"), "Viewing"]
        elif i == best_idx:
            button_color = "outline-success"
            button_text = [html.I(className="fas fa-crown me-1"), "Best"]
        else:
            button_color = "outline-primary"
            button_text = [html.I(className="fas fa-eye me-1"), "View"]
            
        view_button = dbc.Button(
            button_text,
            id={'type': 'enhanced-prompt-view', 'index': i},
            size="sm",
            color=button_color,
            className="btn-sm"
        )
        
        if i == best_idx:  # Highlight best prompt row
            row = html.Tr([
                html.Td([html.I(className="fas fa-crown text-warning me-2"), p], className="fw-bold"),
                html.Td([html.Span(str(positions[i]), className="badge bg-success")]),
                html.Td([html.Span(f"{s:.4f}", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(str(positions[i]), className="badge bg-secondary")),
                html.Td(html.Span(f"{s:.4f}", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with updated buttons
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
            html.Th([html.I(className="fas fa-hashtag me-2"), "Position"], className="bg-light"),
            html.Th([html.I(className="fas fa-chart-line me-2"), "Similarity Score"], className="bg-light"),
            html.Th([html.I(className="fas fa-cogs me-2"), "Actions"], className="bg-light")
        ]), className="thead-light"),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, responsive=True, className="mb-4 shadow-sm")

    # Build enhance results children with updated content
    enhance_children = [
        # Main title with icon
        html.Div([
            html.I(className="fas fa-brain text-primary me-2"),
            html.H5("Enhanced Prompt Analysis", className="d-inline text-primary")
        ], className="mt-3 mb-4"),
        
        # Candidates section
        html.Div([
            html.H6([html.I(className="fas fa-list-alt me-2"), "Candidate Prompts & Scores"], className="mb-3 text-secondary"),
            candidates_table
        ]),
        
        # Current viewing prompt highlight card
        dbc.Alert([
            html.H6([
                html.I(className="fas fa-trophy text-warning me-2") if clicked_index == best_idx else html.I(className="fas fa-eye text-info me-2"),
                "Currently Viewing" if clicked_index != best_idx else "Selected Best Prompt"
            ], className="alert-heading mb-2"),
            html.P(f'"{clicked_prompt}"', className="mb-1 font-monospace"),
            html.Small(f"Position: {clicked_position} | Similarity Score: {clicked_similarity:.4f}", className="text-muted")
        ], color="light" if clicked_index != best_idx else "light", 
           className="border-start border-info border-4" if clicked_index != best_idx else "border-start border-warning border-4"),
        
        # Results section
        html.Div([
            html.H6([html.I(className="fas fa-images me-2"), f"CIR Results for {'Best ' if clicked_index == best_idx else ''}Prompt"], className="mt-4 mb-3 text-secondary"),
            images_grid
        ])
    ]
    
    return enhance_children, enhanced_data

# Callback to populate the prompt enhancement tab with radio options
@callback(
    [Output('prompt-enhancement-content', 'children'),
     Output('prompt-selection', 'options'),
     Output('prompt-selection', 'value')],
    Input('cir-enhanced-prompts-data', 'data'),
    prevent_initial_call=True
)
def populate_prompt_enhancement_tab(enhanced_data):
    """Populate the prompt enhancement tab when new enhanced prompts are available"""
    if not enhanced_data:
        # Clear prompt enhancement tab when starting a new CIR query or no enhancement data
        return [], [], None
    prompts = enhanced_data.get('prompts', [])
    sims = enhanced_data.get('similarities', [])
    positions = enhanced_data.get('positions', [])
    best_idx = enhanced_data.get('best_idx')
    
    # Create styled cards for each enhanced prompt
    cards = []
    for i, (prompt, sim) in enumerate(zip(prompts, sims)):
        is_best = (i == best_idx)
        card_class = "mb-2 border-success" if is_best else "mb-2"
        card_style = {'borderWidth': '2px', 'cursor': 'pointer'} if is_best else {'cursor': 'pointer'}
        icon_class = "fas fa-crown text-warning" if is_best else "fas fa-magic text-info"
        title_class = "mb-1 fw-bold text-success" if is_best else "mb-1 fw-bold"
        title_text = "Best Enhanced Prompt" if is_best else f"Enhanced Prompt {i+1}"
        card = html.Div([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className=f"{icon_class} me-2", style={'fontSize': '1.1em'}),
                        html.H6(title_text, className=title_class),
                        html.P(f'"{prompt}"', className="font-monospace text-dark mb-1", style={'fontSize': '0.9em'}),
                        html.Small([
                            html.Span("Position: ", className="text-muted"),
                            html.Span(str(positions[i]), className="badge bg-secondary ms-1"),
                            html.Br(),
                            html.Span("Similarity Score: ", className="text-muted"),
                            html.Span(f"{sim:.4f}", className="badge bg-secondary ms-1")
                        ])
                    ])
                ])
            ], className=card_class, style=card_style)
        ], id={'type': 'prompt-card', 'index': i}, n_clicks=0)
        cards.append(card)
    
    # Combine radio items for callback wiring (enhanced prompts only)
    all_options = [{'label': f"Enhanced Prompt {i+1}", 'value': i} for i in range(len(prompts))]
    
    content = [
        html.Div([
            html.I(className="fas fa-list-alt text-primary me-2"),
            html.H5("Select Prompt Results to Visualize", className="text-primary mb-3")
        ]),
        html.Div(cards, className="prompt-cards-container")
    ]
    
    return content, all_options, None  # Default to no selection (original CIR)

# Callback to handle prompt card clicks and update selection
@callback(
    [Output('prompt-selection', 'value', allow_duplicate=True),
     Output({'type': 'prompt-card', 'index': ALL}, 'style')],
    Input({'type': 'prompt-card', 'index': ALL}, 'n_clicks'),
    [State('prompt-selection', 'value'),
     State('cir-enhanced-prompts-data', 'data')],
    prevent_initial_call=True
)
def handle_prompt_card_selection(n_clicks_list, current_value, enhanced_data):
    """Handle card clicks for prompt selection"""
    if not any(n_clicks_list) or not enhanced_data:
        raise PreventUpdate
    
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the clicked card index
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        clicked_dict = json.loads(triggered_id)
        clicked_index = clicked_dict.get('index')
    except Exception:
        raise PreventUpdate
    
    # Deselect if clicking the same prompt twice
    if clicked_index == current_value:
        new_selected = -1
    else:
        new_selected = clicked_index
    
    prompts = enhanced_data.get('prompts', [])
    positions = enhanced_data.get('positions', [])
    best_idx = enhanced_data.get('best_idx')
    card_styles = []
    # Enhanced prompt cards styling
    for i in range(len(prompts)):
        is_best = (i == best_idx)
        is_selected = (i == new_selected)
        if is_selected:
            if is_best:
                style = {'backgroundColor': 'rgba(25, 135, 84, 0.1)', 'border': '2px solid #198754'}
            else:
                style = {'backgroundColor': 'rgba(13, 202, 240, 0.1)', 'border': '2px solid #0dcaf0'}
        else:
            style = {}
        card_styles.append(style)
    return new_selected, card_styles

# Callback to update card styles when prompt-selection changes from external sources
@callback(
    Output({'type': 'prompt-card', 'index': ALL}, 'style', allow_duplicate=True),
    Input('prompt-selection', 'value'),
    State('cir-enhanced-prompts-data', 'data'),
    prevent_initial_call=True
)
def update_prompt_card_styles_on_external_change(selected_idx, enhanced_data):
    """Update card styles when prompt-selection changes from external sources (like deselect button)"""
    if not enhanced_data:
        raise PreventUpdate
    
    prompts = enhanced_data.get('prompts', [])
    positions = enhanced_data.get('positions', [])
    best_idx = enhanced_data.get('best_idx')
    card_styles = []
    
    # Enhanced prompt cards styling
    for i in range(len(prompts)):
        is_best = (i == best_idx)
        is_selected = (i == selected_idx)
        
        if is_selected:
            if is_best:
                style = {'backgroundColor': 'rgba(25, 135, 84, 0.1)', 'border': '2px solid #198754'}
            else:
                style = {'backgroundColor': 'rgba(13, 202, 240, 0.1)', 'border': '2px solid #0dcaf0'}
        else:
            style = {}
        
        card_styles.append(style)
    
    return card_styles

# Callback to update all widgets when an enhanced prompt is selected
@callback(
    [Output('gallery', 'children', allow_duplicate=True),
     Output('wordcloud', 'list', allow_duplicate=True),
     Output('histogram', 'figure', allow_duplicate=True),
     Output('scatterplot', 'figure', allow_duplicate=True),
     Output('selected-image-data', 'data', allow_duplicate=True),
     Output('selected-gallery-image-ids', 'data', allow_duplicate=True)],
    Input('prompt-selection', 'value'),
    [State('cir-enhanced-prompts-data', 'data'),
     State('cir-search-data', 'data'),
     State('cir-toggle-state', 'data'),
     State('scatterplot', 'figure'),
     State('selected-gallery-image-ids', 'data')],
    prevent_initial_call=True
)
def update_widgets_for_enhanced_prompt(selected_idx, enhanced_data, search_data, cir_toggle_state, scatterplot_fig, selected_gallery_image_ids):
    """Update widgets for original or enhanced prompt selection, recomputing final query for enhanced and allowing revert."""
    if not cir_toggle_state or selected_idx is None or (enhanced_data is None and selected_idx != -1):
        raise PreventUpdate
    df = Dataset.get()
    # Determine axis and base query coords
    axis_title = scatterplot_fig['layout']['xaxis']['title']['text']
    xq = search_data.get('umap_x_query'); yq = search_data.get('umap_y_query')
    if axis_title != 'umap_x':
        xq = search_data.get('tsne_x_query'); yq = search_data.get('tsne_y_query')
    # Handle original revert
    if selected_idx == -1:
        topk_ids = search_data.get('topk_ids', [])
        top1_id = search_data.get('top1_id')
        xfq = search_data.get('umap_x_final_query'); yfq = search_data.get('umap_y_final_query')
        if axis_title != 'umap_x':
            xfq, yfq = None, None
    else:
        # Enhanced prompt branch
        prompts = enhanced_data.get('prompts', [])
        results_list = enhanced_data.get('all_results', [])
        if selected_idx >= len(results_list):
            raise PreventUpdate
        selected_results = results_list[selected_idx]
        # Build top-k IDs
        topk_ids = []
        for img_name, _ in selected_results:
            try: idx = int(img_name)
            except: idx = img_name
            if idx in df.index:
                topk_ids.append(idx)
        top1_id = topk_ids[0] if topk_ids else None
        # Recompute final query embedding for enhanced prompt
        _, content_string = search_data['upload_contents'].split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(decoded); tmp.close()
        device_model = next(cir_systems.cir_system_searle.clip_model.parameters()).device
        img = Image.open(tmp.name).convert('RGB')
        inp = cir_systems.cir_system_searle.preprocess(img).unsqueeze(0).to(device_model)
        with torch.no_grad():
            feat = cir_systems.cir_system_searle.clip_model.encode_image(inp)
            feat = F.normalize(feat.float(), dim=-1)
        if cir_systems.cir_system_searle.eval_type in ['phi','searle','searle-xl']:
            pseudo = cir_systems.cir_system_searle.phi(feat)
            sel_prompt = prompts[selected_idx]
            cap = f"a photo of $ that {sel_prompt}"
            tok = clip.tokenize([cap], context_length=77).to(device_model)
            from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
            final_feat = encode_with_pseudo_tokens(cir_systems.cir_system_searle.clip_model, tok, pseudo)
            final_feat = F.normalize(final_feat)
        else:
            final_feat = feat
        # Only compute Final Query coordinates for UMAP projection
        if axis_title == 'umap_x':
            umap_path = config.WORK_DIR / 'umap_reducer.pkl'
            umap_reducer = pickle.load(open(str(umap_path),'rb'))
            fnp = final_feat.detach().cpu().numpy()
            fur = umap_reducer.transform(fnp)
            xfq, yfq = float(fur[0][0]), float(fur[0][1])
        else:
            xfq, yfq = None, None  # Final query only shown for UMAP
        os.unlink(tmp.name)
    # Reset CIR traces
    scatterplot_fig['data'] = scatterplot_fig['data'][:3]
    scatterplot_fig['layout']['images'] = []
    main = scatterplot_fig['data'][0]
    xs, ys, cds = main['x'], main['y'], main['customdata']
    
    # Reset main trace colors to remove any previous highlighting from selected images/classes
    main['marker'] = {'color': config.SCATTERPLOT_COLOR}
    # Plot Top-K and Top-1
    x1,y1,xk,yk = [],[],[],[]
    cmp1 = int(top1_id) if top1_id is not None else None
    cmpk = [int(i) for i in topk_ids]
    for xi, yi, val in zip(xs, ys, cds):
        try: v = int(val)
        except: v = val
        if v==cmp1: x1.append(xi); y1.append(yi)
        elif v in cmpk: xk.append(xi); yk.append(yi)
    if xk:
        scatterplot_fig['data'].append(go.Scatter(x=xk,y=yk,mode='markers',marker=dict(color=config.TOP_K_COLOR,size=7),name='Top-K').to_plotly_json())
    if x1:
        scatterplot_fig['data'].append(go.Scatter(x=x1,y=y1,mode='markers',marker=dict(color=config.TOP_1_COLOR,size=9),name='Top-1').to_plotly_json())
    # Plot Query and Final Query
    if xq is not None:
        scatterplot_fig['data'].append(go.Scatter(x=[xq],y=[yq],mode='markers',marker=dict(color=config.QUERY_COLOR,size=12,symbol='star'),name='Query').to_plotly_json())
    if xfq is not None:
        scatterplot_fig['data'].append(go.Scatter(x=[xfq],y=[yfq],mode='markers',marker=dict(color=config.FINAL_QUERY_COLOR,size=10,symbol='diamond'),name='Final Query').to_plotly_json())
    # Wordcloud
    counts = df.loc[topk_ids]['class_name'].value_counts()
    if len(counts):
        wg = wordcloud.wordcloud_weight_rescale(counts.values,1,counts.max())
        wc = sorted([[cn,w] for cn,w in zip(counts.index,wg)],key=lambda x:x[1],reverse=True)
    else: wc=[]
    # Gallery and Histogram
    cir_df = df.loc[topk_ids]
    # Clear any previous gallery selections when switching to enhanced prompt results
    gal = gallery.create_gallery_children(cir_df['image_path'].values,cir_df['class_name'].values,cir_df.index.values,[])
    hist = histogram.draw_histogram(cir_df)
    return gal, wc, hist, scatterplot_fig, None, []

@callback(
    Output('model-change-flag', 'children'),
    Output('cir-results', 'children', allow_duplicate=True),
    Output('cir-toggle-button', 'children', allow_duplicate=True),
    Output('cir-toggle-button', 'color', allow_duplicate=True),
    Output('cir-toggle-button', 'style', allow_duplicate=True),
    Output('cir-toggle-state', 'data', allow_duplicate=True),
    Output('cir-run-button', 'style', allow_duplicate=True),
    Input('custom-dropdown', 'value'),
    prevent_initial_call=True
)
def clear_results_on_model_change(_):
    return (
        "changed",
        html.Div("Model changed. Please run a new search.", className="text-muted text-center p-4"),
        'Visualize CIR results',
        'success',
        {'display': 'none', 'color': 'black'},
        False,
        {'display': 'block', 'color': 'black'}
    )