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
# Ensure the SEARLE demo directory is on the path for local imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'SEARLE'))
from compose_image_retrieval_demo import ComposedImageRetrievalSystem
from src.Dataset import Dataset
import torch
import torch.nn.functional as F
import pickle
import clip
from dash.exceptions import PreventUpdate
from dash import ALL
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# Thread lock and CIR system instance
lock = threading.Lock()
cir_system = None

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
     Output('cir-enhanced-prompts-data', 'data', allow_duplicate=True)],
    [Input('cir-search-button', 'n_clicks')],
    [State('cir-upload-image', 'contents'), State('cir-text-prompt', 'value'), State('cir-top-n', 'value')],
    prevent_initial_call=True
)
def perform_cir_search(n_clicks, upload_contents, text_prompt, top_n):
    """Perform CIR search using the SEARLE ComposedImageRetrievalSystem"""
    if not upload_contents or not text_prompt:
        empty = html.Div("No results yet. Upload an image and enter a text prompt to start retrieval.", className="text-muted text-center p-4")
        # Show Run CIR button, hide visualize button, clear enhance results and data
        return empty, html.Div(), None, {'display': 'none', 'color': 'black'}, {'display': 'block', 'color': 'black'}, [], None
    try:
        # Decode and save query image
        _, content_string = upload_contents.split(',')
        decoded = base64.b64decode(content_string)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        tmp.write(decoded)
        tmp.close()

        global cir_system
        if cir_system is None:
            with lock:
                if cir_system is None:
                    cir_system = ComposedImageRetrievalSystem(
                        dataset_path=config.CIR_DATASET_PATH,
                        dataset_type=config.CIR_DATASET_TYPE,
                        clip_model_name=config.CIR_CLIP_MODEL_NAME,
                        eval_type=config.CIR_EVAL_TYPE,
                        preprocess_type=config.CIR_PREPROCESS_TYPE,
                        exp_name=config.CIR_EXP_NAME,
                        phi_checkpoint_name=config.CIR_PHI_CHECKPOINT_NAME,
                        features_path=config.CIR_FEATURES_PATH,
                        load_features=config.CIR_LOAD_FEATURES,
                    )
                    cir_system.create_database(split=config.CIR_SPLIT)

        with lock:
            results = cir_system.query(tmp.name, text_prompt, top_n)

        # Build result cards using paths from the loaded dataset
        cards = []
        df = Dataset.get()
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
                continue
            data = open(img_path, 'rb').read()
            src = f"data:image/jpeg;base64,{base64.b64encode(data).decode()}"
            
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

        rows = []
        for i in range(0, len(cards), 5):
            chunk = cards[i:i+5]
            cols = [dbc.Col(c, width=12//len(chunk)) for c in chunk]
            rows.append(dbc.Row(cols, className='mb-3'))

        results_div = html.Div([html.H5("Retrieved Images", className="mb-3")] + rows)
        status = html.Div([html.I(className="fas fa-check-circle text-success me-2"), f"Retrieved {len(cards)} images"], className="text-success small")
        # Prepare store data for visualization
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
        # Compute query embedding and normalize
        device_model = next(cir_system.clip_model.parameters()).device
        query_img = Image.open(tmp.name).convert('RGB')
        query_input = cir_system.preprocess(query_img).unsqueeze(0).to(device_model)
        with torch.no_grad():
            feat = cir_system.clip_model.encode_image(query_input)
            feat = F.normalize(feat.float(), dim=-1)
        feat_np = feat.cpu().numpy()
        
        # Compute the final composed query embedding (image + text) used for actual search
        with torch.no_grad():
            if cir_system.eval_type in ['phi', 'searle', 'searle-xl']:
                # Use phi network to generate pseudo tokens
                pseudo_tokens = cir_system.phi(feat)
                # Create text with pseudo token placeholder
                input_caption = f"a photo of $ that {text_prompt}"
                tokenized_caption = clip.tokenize([input_caption], context_length=77).to(device_model)
                # Encode text with pseudo tokens - this is the final query embedding
                from src.callbacks.SEARLE.src.encode_with_pseudo_tokens import encode_with_pseudo_tokens
                final_query_features = encode_with_pseudo_tokens(
                    cir_system.clip_model, tokenized_caption, pseudo_tokens
                )
                final_query_features = F.normalize(final_query_features)
            else:
                # For other eval types, use the reference image features as fallback
                final_query_features = feat
        
        final_query_feat_np = final_query_features.cpu().numpy()
        
        # UMAP transform
        umap_path = config.WORK_DIR / 'umap_reducer.pkl'
        umap_reducer = pickle.load(open(str(umap_path), 'rb'))
        umap_xy = umap_reducer.transform(feat_np)
        umap_x_query, umap_y_query = float(umap_xy[0][0]), float(umap_xy[0][1])
        
        # UMAP transform for final composed query
        final_umap_xy = umap_reducer.transform(final_query_feat_np)
        umap_x_final_query, umap_y_final_query = float(final_umap_xy[0][0]), float(final_umap_xy[0][1])
        
        # Delete temporary query image file
        os.unlink(tmp.name)
        store_data = {
            'topk_ids': topk_ids,
            'top1_id': top1_id,
            'umap_x_query': umap_x_query,
            'umap_y_query': umap_y_query,
            'umap_x_final_query': umap_x_final_query,
            'umap_y_final_query': umap_y_final_query,
            'tsne_x_query': None,  # Not used since Query is only shown for UMAP
            'tsne_y_query': None,  # Not used since Query is only shown for UMAP
            'upload_contents': upload_contents,
            'text_prompt': text_prompt,
            'top_n': top_n
        }
        # Show visualize button, hide Run CIR, clear enhance data
        return results_div, status, store_data, {'display': 'block', 'color': 'black'}, {'display': 'none', 'color': 'black'}, [], None
    except Exception as e:
        err = html.Div([html.I(className="fas fa-exclamation-triangle text-danger me-2"), f"Retrieval error: {e}"], className="text-danger small")
        # On error, hide visualize button, show Run CIR, clear enhance results and data
        return html.Div("Error occurred during image retrieval.", className="text-danger text-center p-4"), err, None, {'display': 'none', 'color': 'black'}, {'display': 'block', 'color': 'black'}, [], None

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
        temperature=0.8,
        top_p=0.9,
        num_return_sequences=N,
    )
    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # Extract enhanced prompts from between <ANSWER> tags (case-insensitive)
    prompts = []
    for i, result in enumerate(results):
        
        # First extract only the response part after [/INST]
        inst_idx = result.find('[/INST]')
        if inst_idx == -1:
            print(f"No [/INST] found in result {i+1}")
            continue
        
        response_part = result[inst_idx + 7:]  # Skip '[/INST]'
        
        # Now look for ANSWER tags in the response part only
        start_tag = '<ANSWER>'
        end_tag = '</ANSWER>'
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
    
    print(f"Final prompts list: {prompts}")

    # Prepare ideal image embedding
    df = Dataset.get()
    try:
        ideal_idx = int(selected_image_id)
    except:
        ideal_idx = selected_image_id
    ideal_path = df.loc[ideal_idx]['image_path']
    ideal_img = Image.open(ideal_path).convert('RGB')
    device_model = next(cir_system.clip_model.parameters()).device
    ideal_input = cir_system.preprocess(ideal_img).unsqueeze(0).to(device_model)
    with torch.no_grad():
        ideal_feat = cir_system.clip_model.encode_image(ideal_input)
        ideal_feat = F.normalize(ideal_feat.float(), dim=-1).squeeze(0)

    # Score each candidate prompt and store full results
    sims = []
    all_prompt_results = []
    for p in prompts:
        # Use the same query method as full CIR to get full results
        full_prompt_results = cir_system.query(tmp.name, p, search_data['top_n'])
        all_prompt_results.append(full_prompt_results)
        
        # Find the similarity score for the ideal image in these results
        ideal_score = None
        for name, score in full_prompt_results:
            if str(name) == str(selected_image_id):
                ideal_score = score
                break
        
        # If ideal image not found in top results, assign very low score
        if ideal_score is None:
            ideal_score = 0.0
            print(f"Warning: Ideal image {selected_image_id} not found in top-{search_data['top_n']} for prompt: {p}")
        
        sims.append(ideal_score)

    # Select best prompt by highest similarity
    best_idx = sims.index(max(sims))
    best_prompt = prompts[best_idx]
    best_sim_score = sims[best_idx]
    print(f"Selected best prompt: '{best_prompt}' with score: {best_sim_score}")

    # Get results for best prompt (already computed)
    full_results = all_prompt_results[best_idx]
    
    # Clean up temporary file
    os.unlink(tmp.name)

    # Status message with icon
    status = html.Div([
        html.I(className="fas fa-magic text-success me-2"),
        "Enhanced prompt generated successfully! See analysis below."
    ], className="text-success small mb-3")

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
                html.Td([html.Span(f"{s:.4f}", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(f"{s:.4f}", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with better styling and action column
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
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
            html.Small(f"Similarity Score: {best_sim_score:.4f}", className="text-muted")
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
    all_results = enhanced_data['all_results']
    best_idx = enhanced_data['best_idx']
    
    clicked_prompt = prompts[clicked_index]
    clicked_similarity = similarities[clicked_index]
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
                html.Td([html.Span(f"{s:.4f}", className="badge bg-success")]),
                html.Td(view_button)
            ], className="table-success")
        else:
            row = html.Tr([
                html.Td(p),
                html.Td(html.Span(f"{s:.4f}", className="badge bg-secondary")),
                html.Td(view_button)
            ])
        table_rows.append(row)

    # Enhanced candidates table with updated buttons
    candidates_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th([html.I(className="fas fa-edit me-2"), "Generated Prompts"], className="bg-light"),
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
            html.Small(f"Similarity Score: {clicked_similarity:.4f}", className="text-muted")
        ], color="light" if clicked_index != best_idx else "light", 
           className="border-start border-info border-4" if clicked_index != best_idx else "border-start border-warning border-4"),
        
        # Results section
        html.Div([
            html.H6([html.I(className="fas fa-images me-2"), f"CIR Results for {'Best ' if clicked_index == best_idx else ''}Prompt"], className="mt-4 mb-3 text-secondary"),
            images_grid
        ])
    ]
    
    return enhance_children, enhanced_data