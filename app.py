import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import gradio as gr

# --- 1. Define Model Architecture ---

# --- Hyperparameters ---
VOCAB_SIZE = 50258
EMBED_DIM = 512
MAX_SEQ_LEN = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "final_model.pth"

# --- Model Classes (Unchanged) ---
class TokenAndPositionalEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_embedding(x)
        pos = torch.arange(0, T, dtype=torch.long, device=DEVICE)
        pos_emb = self.positional_embedding(pos)
        x = tok_emb + pos_emb
        return self.dropout(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        wei = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        wei = wei.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, max_seq_len)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers):
        super().__init__()
        self.embedding_layer = TokenAndPositionalEmbedding(
            vocab_size, embed_dim, max_seq_len
        )
        self.blocks = nn.ModuleList(
            [DecoderBlock(embed_dim, num_heads, max_seq_len) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
    def forward(self, x):
        if x.shape[1] > MAX_SEQ_LEN:
            x = x[:, -MAX_SEQ_LEN:]
        x = self.embedding_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# --- 2. Load Tokenizer & Model Weights ---
print("Loading tokenizer and model...")

tokenizer = GPT2Tokenizer.from_pretrained(
    "cahya/gpt2-small-indonesian-522M", 
    use_fast=False
)
special_tokens_dict = {'eos_token': '<|endoftext|>', 'pad_token': '<|pad|>'}
tokenizer.add_special_tokens(special_tokens_dict)

model = GPT(
    VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN, NUM_HEADS, NUM_LAYERS
).to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    print("...Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# --- 3. --- CORRECTED: Streaming Generation Function ---
# This version is updated to work with Gradio's `type="messages"` format

def predict(message, history, max_len, temp, top_p, rep_pen):
    """
    'history' is now a List of Dictionaries:
    [
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'}
    ]
    """
    
    # --- THIS IS THE FIX ---
    # 1. Append the user's new message to the history first
    history.append({"role": "user", "content": message})
    # --- END OF FIX ---

    # 2. Format the prompt (including history)
    full_prompt = ""
    for turn in history:
        if turn['role'] == 'user':
            full_prompt += f"User: {turn['content']}\n\nModel: "
        else: # 'assistant'
            # Only add content if the assistant has said something
            if turn['content']:
                full_prompt += f"{turn['content']}{tokenizer.eos_token}\n"
    
    # Add the "Model: " prefix for the new generation
    full_prompt += f"Model: "
    
    # 3. Tokenize
    prompt_ids = tokenizer(full_prompt, return_tensors="pt")['input_ids'].to(DEVICE)
    
    # 4. Add a placeholder for the bot's response
    history.append({"role": "assistant", "content": ""})
    
    # 5. Generate response
    generated_ids = prompt_ids
    full_response = ""
    
    with torch.no_grad():
        for _ in range(max_len):
            
            current_token_len = generated_ids.shape[1]
            if current_token_len > MAX_SEQ_LEN:
                generated_ids = generated_ids[:, - (MAX_SEQ_LEN - 1):]
            
            logits = model(generated_ids)[:, -1, :]
            
            # Apply all sampling settings
            logits = logits / temp
            if rep_pen > 1.0:
                recent_tokens = generated_ids[0, -50:]
                logits[0].scatter_add_(0, recent_tokens, torch.full_like(recent_tokens, -rep_pen, dtype=torch.float))
            
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')

            next_token_probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1)

            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break
                
            # --- Streaming logic ---
            new_token = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            full_response += new_token
            
            # Update the history's last message
            history[-1]['content'] = full_response
            
            # Yield the updated history to Gradio
            yield history

# --- 4. Launch the Gradio Website with Sliders ---
print("Launching Gradio UI...")

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Chatbot Pendongeng Handal 😇")
    gr.Markdown("Trained from scratch dari dataset RP Bahasa Indonesia.")
    
    with gr.Row():
        # Left column for settings
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Generation Settings")
            max_len_slider = gr.Slider(
                minimum=10, maximum=500, value=100, step=10,
                label="Max Output Length"
            )
            temp_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="Temperature"
            )
            top_p_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                label="Top-p (Nucleus Sampling)"
            )
            rep_pen_slider = gr.Slider(
                minimum=1.0, maximum=2.0, value=1.2, step=0.1,
                label="Repetition Penalty"
            )
        
        # Right column for the chat
        with gr.Column(scale=4):
            # --- FIX 1: Removed 'label="Chat History"' ---
            chatbot = gr.Chatbot(height=500, type="messages")
            
            # --- FIX 2: Create a row for Textbox and Send Button ---
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message", 
                    placeholder="Ketik di sini...", 
                    scale=7, # Make textbox larger
                    container=False # Remove container background for cleaner look
                )
                submit_btn = gr.Button(
                    "Send", 
                    variant="primary", 
                    scale=1 # Make button smaller
                )
            
            # --- FIX 3: Removed the 'clear = gr.ClearButton(...)' line ---
            # The built-in trash icon in the chatbot window (top-right)
            # already clears the chat.

    # --- FIX 4: Hook up both Enter key (msg.submit) and Send button (submit_btn.click) ---
    
    # Define the function to call on submit
    # We use a list to hold the 'msg' so we can update it
    submit_action = msg.submit(
        fn=predict, 
        inputs=[msg, chatbot, max_len_slider, temp_slider, top_p_slider, rep_pen_slider], 
        outputs=[chatbot]
    ).then(
        fn=lambda: gr.update(value=""), # Clear the textbox
        inputs=None,
        outputs=[msg],
        queue=False
    )
    
    # Also trigger the same actions when the Send button is clicked
    submit_btn.click(
        fn=predict, 
        inputs=[msg, chatbot, max_len_slider, temp_slider, top_p_slider, rep_pen_slider], 
        outputs=[chatbot]
    ).then(
        fn=lambda: gr.update(value=""), # Clear the textbox
        inputs=None,
        outputs=[msg],
        queue=False
    )

# launch() will start the web server
iface.launch()


