import modules.scripts as scripts
import gradio as gr
import os
import pickle
import torch
import torch.nn as nn
import open_clip
import mediafire_dl
from torch.nn import functional as nnf
from modules import script_callbacks
from transformers import  GPT2Tokenizer, GPT2LMHeadModel, pipeline, AutoModelForMaskedLM, AutoTokenizer
from typing import List, Optional, Union, Tuple, Dict, Any
from itertools import combinations

original_directory = os.getcwd()
current_directory = os.path.join(original_directory,'extensions','ICG_sd_extension','scripts')
previous_choice = 'ClipCap'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encyclopedia, labels = None, None
modelCLIP, preprocess, tokenizerCLIP, model, tokenizer, generator = None, None, None, None, None, None
text_features_1000, prime_labels, prime_text_features = None, None, None
prime_check_list = ['landscape','people','animal','plant','some object']
prime_mapping = {'landscape':'scenery','people':'human','animal':'animal','plant':'plant','some object':'object'} # need mapping because the wordings in label classification is different from those of CLIP's
secondary_check_list = ['time','weather','activity']
prefix_length = 10

secondary_text_labels_landscape = None
secondary_text_labels_human = None
secondary_text_labels_occupation = None
secondary_text_labels_animal = None
secondary_text_labels_plant = None
secondary_text_labels_object = None
secondary_text_labels_activity = None
secondary_text_labels_time = None
secondary_text_labels_weather = None
secondary_text_labels_clothing = None

secondary_text_features_landscape = None
secondary_text_features_human = None
secondary_text_features_occupation = None
secondary_text_features_animal = None
secondary_text_features_plant = None
secondary_text_features_object = None
secondary_text_features_activity = None
secondary_text_features_time = None
secondary_text_features_weather = None
secondary_text_features_clothing = None



def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row():
            with gr.Column(variant='panel'):
                image = gr.Image(type='pil', label="Image", source='upload')
            with gr.Column(variant='panel'):
                    
                choice = gr.Radio(choices = ["ClipCap", "Selective Blind Guessing"], value = 'ClipCap', label="Model")
                
                btn = gr.Button("Generate caption", variant='primary').style(
                    full_width=False
                    )
                
                caption_display = gr.Textbox(
                    default="",
                    label="Caption",
                    interactive = False
                    )

        btn.click(
            get_caption,
            inputs = [choice, image],
            outputs = caption_display,
        )

        return [(ui_component, "Image Caption", "image_caption_tab")]

def get_caption(choice, image = None):
    global current_directory, original_directory
    global previous_choice
    global device, encyclopedia, labels, modelCLIP, preprocess, tokenizerCLIP, model, tokenizer, generator
    global text_features_1000, prime_labels, prime_text_features, prime_check_list, prime_mapping, secondary_check_list
    global secondary_text_labels_landscape, secondary_text_labels_human, secondary_text_labels_occupation, secondary_text_labels_animal, secondary_text_labels_plant, secondary_text_labels_object, secondary_text_labels_activity, secondary_text_labels_time, secondary_text_labels_weather, secondary_text_labels_clothing
    global secondary_text_features_landscape, secondary_text_features_human, secondary_text_features_occupation, secondary_text_features_animal, secondary_text_features_plant, secondary_text_features_object, secondary_text_features_activity, secondary_text_features_time, secondary_text_features_weather, secondary_text_features_clothing
    global prefix_length
    if image is not None:
        torch.cuda.empty_cache()
        
        # things to always have value, they aren't too heavy
        # load encyclopedia
        if encyclopedia is None:
            with open(os.path.join(current_directory,'encyclopedia.pkl'), 'rb') as f:
              encyclopedia = pickle.load(f)
        # load labels
        if labels is None:
            with open(os.path.join(current_directory,'all_labels.pkl'), 'rb') as f:
              labels = pickle.load(f)
              
            labels = [keyword for keyword, keyword_pos, frequency in labels if keyword_pos in ['NOUN','ADV','ADJ']]
            labels = labels[:1000]
        
        # things that need to be re-declare if changed, they are heavy
        if choice != previous_choice:
            previous_choice = choice
            change_model = True
        else:
            change_model = False
            
        if change_model or model is None:
            change_model = False
            if choice == 'ClipCap':
                model_path = os.path.join(current_directory,'flickr8k_prefix-030.pt')
                modelCLIP, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device= device, precision= 'fp16' if device == torch.device('cuda') else 'fp32')              
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                # Load model weights
                model = ClipCaptionModel(prefix_length)
                if not os.path.exists(model_path):
                    url = 'https://download1649.mediafire.com/0jzo588ip91gq4T9uNZPEPaAJrIf8srGL3PIZ5kDUCpOsMHFjn-8xr4786EaZGA8IUxowMuUy3UnHc4aKGlWZHQBL7R4rArvkbhY_pzIuZyv_rGjA36yV38WAPkplghdo11g44kF5LBFvcLSp4dMHqdXUw4WWi2JnfRP03l7sTONEg/qof8qa7odm4dfck/flickr8k_prefix-030.pt'
                    mediafire_dl.download(url, model_path, quiet=False)
                    
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                model = model.eval()
                model = model.to(device)
            else:              
                # initiate models
                modelCLIP, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device= device, precision= 'fp16' if device == torch.device('cuda') else 'fp32')
                tokenizerCLIP = open_clip.get_tokenizer('ViT-L-14')
                tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                tokenizer.sep_token = '###' # set separator
                tokenizer.pad_token = tokenizer.eos_token # set padding token
                model_name = 'top_hierarchy'
                model_path = os.path.join(current_directory, model_name, 'model.safetensors')
                if not os.path.exists(model_path):
                    url = 'https://download1323.mediafire.com/u0312q9aa6lgEsgJmP1dLaDF0p7Td3f4fnWzlRH_NXvgPtskQq_GpRL23mZ50cv_R-0Ls31a9cvqgX1lUvBevqDDfNrMVNHzJ_FDg0B-81Whaslf5NatrZ4-exVjkuQeBu3EKd8LO0FFiUSsveSoaEZ88M8aWaA_XWFibLMa-Moszg/9rjol6786rlmefx/model.safetensors'
                    mediafire_dl.download(url, model_path, quiet=False)
                    
                os.chdir(current_directory)
                generator = pipeline("text-generation", model=model_name, tokenizer=tokenizer)
                os.chdir(original_directory)
                
                # Create all the text features
                text_features_1000 = get_text_features(labels=labels, model=modelCLIP, tokenizer=tokenizerCLIP)
                prime_labels = get_prime_labels()
                prime_text_features = get_text_features(labels=prime_labels, model=modelCLIP, tokenizer=tokenizerCLIP)
                prime_check_list = ['landscape','people','animal','plant','some object']
                prime_mapping = {'landscape':'scenery','people':'human','animal':'animal','plant':'plant','some object':'object'} # need mapping because the wordings in label classification is different from those of CLIP's
                secondary_check_list = ['time','weather','activity']

                secondary_text_labels_landscape = get_secondary_labels('scenery')
                secondary_text_labels_human = get_secondary_labels('human')
                secondary_text_labels_occupation = get_secondary_labels('occupation')
                secondary_text_labels_animal = get_secondary_labels('animal')
                secondary_text_labels_plant = get_secondary_labels('plant')
                secondary_text_labels_object = get_secondary_labels('object')
                secondary_text_labels_activity = get_secondary_labels('activity')
                secondary_text_labels_time = get_secondary_labels('time')
                secondary_text_labels_weather = get_secondary_labels('weather')
                secondary_text_labels_clothing = get_secondary_labels('clothing')

                secondary_text_features_landscape = get_text_features(labels=secondary_text_labels_landscape, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_human = get_text_features(labels=secondary_text_labels_human, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_occupation = get_text_features(labels=secondary_text_labels_occupation, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_animal = get_text_features(labels=secondary_text_labels_animal, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_plant = get_text_features(labels=secondary_text_labels_plant, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_object = get_text_features(labels=secondary_text_labels_object, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_activity = get_text_features(labels=secondary_text_labels_activity, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_time = get_text_features(labels=secondary_text_labels_time, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_weather = get_text_features(labels=secondary_text_labels_weather, model=modelCLIP, tokenizer=tokenizerCLIP)
                secondary_text_features_clothing = get_text_features(labels=secondary_text_labels_clothing, model=modelCLIP, tokenizer=tokenizerCLIP)
        
        caption = ''
        
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast():
            img_features = modelCLIP.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        
        if choice == 'ClipCap':
            with torch.no_grad(), torch.cuda.amp.autocast():
                prefix_embed = model.clip_project(img_features.to(dtype=torch.float32)).reshape(1, prefix_length, -1)
            
            caption = generate_clip_prefix(model, tokenizer, embed=prefix_embed).split('.')[0].strip()+' .'
        else:        
            caption = generate_caption(img_features, 5)
        
        return caption
        
    else:
        return ''
        
def get_top_percent(label_dict: dict[str,float], threshold: int = 0.5) -> List[str]:
  '''
    label_dict: dict of probability
    threshold: filter
  '''

  current = 0
  labels = []

  for label, prob in label_dict.items():
    current += prob
    labels.append(label)
    if current > threshold:
      break
  return labels
  
def get_text_features(labels: List[str], model, tokenizer, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.Tensor:
  # tokenize
  text_tokens = tokenizer(labels).to(device)
  # get features
  with torch.no_grad(), torch.cuda.amp.autocast():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

  return text_features

def get_prob_dict(img_features: torch.Tensor, text_features: torch.Tensor, labels: List[str], sort: bool = False) -> Dict[str,float]:
  # Get probability
  text_probs = (100.0 * img_features @ text_features.T).softmax(dim=-1).tolist()[0]
  # Get dictionary
  prob_dict = {}
  for x in range(len(labels)):
    prob_dict[labels[x]] = text_probs[x]
  if sort:
    prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse = True))

  return prob_dict
  
  
def get_prime_labels() -> List[str]:
  prime_subject = ['landscape','people','animal','plant','some object']
  labels = []
  labels+= ['picture of '+ a +'.' for a in prime_subject]
  labels+= ['picture of '+ a[0] + ' and ' + a[1] + '.' for a in combinations(prime_subject,2)]
  labels+= ['picture of '+ a[0] + ' and ' + a[1] + ' and ' + a[2] + '.' for a in combinations(prime_subject,3)]

  return labels

def get_secondary_labels(mode: str) -> List[str]:
  '''
    mode: human, scenery, plant, object, activity, weather, time
  '''
  # human get more attention because most description in dataset feature human subject
  if mode == 'human':
    subject = ['a man','a woman','a boy','a girl','an old man','an old woman']
    subjects = ['men','women', 'boys','girls','old men','old women']
    count = ['two','three','four']

    subjects_count = [] # people with count number
    labels = [f'picture of {a}.' for a in subject]

    for a in count:
      for b in subjects:
        subjects_count.append(f'{a} {b}')

    labels += [f'picture of {a}.' for a in subjects_count]
    labels += [f'picture of {a} people.' for a in count]
    labels.append('picture of a crowd of people.')
  elif mode == 'occupation':
    labels = [f'picture of {a}.' for a in encyclopedia['human']]
  elif mode == 'scenery':
    labels = [f'picture of {a}.' for a in encyclopedia['scenery'].keys()]
  elif mode == 'animal':
    labels = [f'picture of {a}.' for a in encyclopedia['animal'].keys()]
  elif mode == 'plant':
    labels = [f'picture of {a}.' for a in encyclopedia['plant'].keys()]
  elif mode == 'object':
    labels = [f'picture of {a}.' for a in encyclopedia['object'].keys() if a != 'clothing and fashion']
  elif mode == 'activity':
    labels = [f'activity is {a}.' for a in encyclopedia['activity'].keys()]
  elif mode == 'time':
    labels = [f'picture during {a}.' for a in encyclopedia['time'].keys()]
  elif mode == 'weather':
    labels = [f'picture during {a}.' for a in encyclopedia['weather'].keys()]
  elif mode == 'clothing':
    color = ['red', 'orange','yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'white', 'black']
    labels = [f'wearing {a} shirt.' for a in color]
    labels += [f'wearing {a} suit.' for a in color]
    labels += [f'wearing {a} cloth.' for a in color]
    labels += [f'wearing {a} jacket.' for a in color]
  else:
    raise Exception(f"\033[1;31;40m  Unsupported mode for labeling: {mode}")
  return labels

def get_tertiary_labels(mode: str, sub_mode: str, isverb: bool = False) -> List[str]:
  if not isverb:
    return [f'picture of {a}.' for a in encyclopedia[mode][sub_mode]]
  else:
    return [f'activity is {a}.' for a in encyclopedia[mode][sub_mode]]

def generate_caption(img_features: torch.Tensor, generation_count: int = 5):
  # prob dict for 1000 keywords
  prob_dict = get_prob_dict(img_features=img_features, text_features=text_features_1000, labels=labels, sort=True)
  # list of keywords for caption
  main_list = get_top_percent(prob_dict)

  # prob dict for hierarchy selection
  prime_prob_dict = get_prob_dict(img_features=img_features, text_features=prime_text_features, labels=prime_labels, sort=True)
  prime = list(prime_prob_dict.keys())[0] + ' landscape some object' # guaranteed landscape and object

  # add keywords through hierarchy selection
  for a in prime_check_list:
    if a in prime:
      if a == 'landscape':
        secondary_labels = secondary_text_labels_landscape
        text_features = secondary_text_features_landscape
      elif a == 'people':
        secondary_labels = secondary_text_labels_human
        text_features = secondary_text_features_human
      elif a == 'animal':
        secondary_labels = secondary_text_labels_animal
        text_features = secondary_text_features_animal
      elif a == 'plant':
        secondary_labels = secondary_text_labels_plant
        text_features = secondary_text_features_plant
      elif a == 'some object':
        secondary_labels = secondary_text_labels_object
        text_features = secondary_text_features_object
      else:
        raise Exception(f"\033[1;31;40m  Unsupported labeling: {a}")

      secondary_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=secondary_labels, sort=True)
      secondary = list(secondary_prob_dict.keys())

      if a == 'people':
        main_list.append(secondary[0].split('picture of')[1].strip()[:-1])

        occupation_labels = secondary_text_labels_occupation
        text_features = secondary_text_features_occupation
        occupation_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=occupation_labels, sort=True)
        occupation = list(occupation_prob_dict.keys())
        main_list.append(occupation[0].split('picture of')[1].strip()[:-1])

        clothing_labels = secondary_text_labels_clothing
        text_features = secondary_text_features_clothing
        clothing_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=clothing_labels, sort=True)
        clothing = list(clothing_prob_dict.keys())
        main_list.append(clothing[0].split('wearing')[1].strip()[:-1])

      else:
        sub_mode = secondary[0].split('picture of')[1].strip()[:-1]
        main_list.append(sub_mode)
        try: # tertiary labels may not exist
          tertiary_labels = get_tertiary_labels(prime_mapping[a], sub_mode)[:100]
          text_features = get_text_features(labels=tertiary_labels, model=modelCLIP, tokenizer=tokenizerCLIP)
          tertiary_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=tertiary_labels, sort=True)
          tertiary = list(tertiary_prob_dict.keys())

          # prime subject of human and animal only use the highest probability, the rest use 3 highest probabilities
          if a == 'animal':
            main_list.append(tertiary[0].split('picture of')[1].strip()[:-1])
          else:
            main_list += [a.split('picture of')[1].strip()[:-1] for a in tertiary[:3]]
        except: pass

  for a in secondary_check_list:
    if not ((a == 'weather' and 'indoor' in main_list) or (a == 'activity' and 'people' not in prime and 'animal' not in prime)):
      if a == 'activity':
        secondary_labels = secondary_text_labels_activity
        text_features = secondary_text_features_activity
      elif a == 'time':
        secondary_labels = secondary_text_labels_time
        text_features = secondary_text_features_time
      elif a == 'weather':
        secondary_labels = secondary_text_labels_weather
        text_features = secondary_text_features_weather
      else:
        raise Exception(f"\033[1;31;40m  Unsupported labeling: {a}")

      secondary_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=secondary_labels, sort=True)
      secondary = list(secondary_prob_dict.keys())
      if a != 'activity':
        main_list.append(secondary[0].split('picture during')[1].strip()[:-1])
      else:
        sub_mode = secondary[0].split('activity is')[1].strip()[:-1]
        try: # tertiary labels may not exist
          tertiary_labels = get_tertiary_labels(a, sub_mode, True) [:100]
          text_features = get_text_features(labels=tertiary_labels, model=modelCLIP, tokenizer=tokenizerCLIP)
          tertiary_prob_dict = get_prob_dict(img_features=img_features, text_features=text_features, labels=tertiary_labels, sort=True)
          tertiary = list(tertiary_prob_dict.keys())
          main_list.append(tertiary[0].split('activity is')[1].strip()[:-1])
        except: pass

  # sort keywords by probability
  final_text_features = get_text_features(labels=main_list, model=modelCLIP, tokenizer=tokenizerCLIP)
  final_keywords = list(get_prob_dict(img_features=img_features, text_features=final_text_features, labels=main_list, sort=True).keys())

  # Generate caption --------------------------------------------

  if generation_count < 1:
    generation_count = 1

  prompt = ', '.join(final_keywords) + ' ' + tokenizer.sep_token
  best_result = generator(prompt, max_length = 256, pad_token_id=tokenizer.eos_token_id)[0]['generated_text'].split(tokenizer.sep_token)[1].strip().split('.')[0].strip() + ' .'
  if generation_count != 1:
    results = []
    results.append(best_result)
    for z in range(generation_count - 1):
        result = generator(prompt, max_length = 256, pad_token_id=tokenizer.eos_token_id)[0]['generated_text'].split(tokenizer.sep_token)[1].strip().split('.')[0].strip() + ' .'
        results.append(result)

    text_features = get_text_features(results, modelCLIP, tokenizerCLIP)
    # compare best result
    best_result = list(get_prob_dict(img_features, text_features, results, True).keys())[0]

  return best_result
  


# clip prefix -----------------------------------------------------------------------
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class ClipCaptionModel(nn.Module):

    #@functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        #print(embedding_text.size()) #torch.Size([5, 67, 768])
        #print(prefix_projections.size()) #torch.Size([5, 1, 768])

        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, prefix_size: int = 768, tokenizer_type: str = 'gpt2'):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt.resize_token_embeddings(len(GPT2Tokenizer.from_pretrained(tokenizer_type)))
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))

# Caption prediction
def generate_clip_prefix(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]
script_callbacks.on_ui_tabs(on_ui_tabs)
