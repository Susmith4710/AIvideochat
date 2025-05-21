# Video Analysis with Qwen2-VL for Google Cloud with T4 GPU
# This notebook allows you to analyze YouTube videos or uploaded videos using the Qwen2-VL model

#update Gradio to the latest version for proper component support if you haven't
!pip install -q --upgrade gradio

# Install other required packages
!pip install -q transformers
!pip install -q qwen-vl-utils
!pip install -q yt-dlp
!pip install -q --no-build-isolation flash-attn
!pip install -q accelerate
!pip install -q huggingface_hub
!pip install -q einops
!pip install -q bitsandbytes  # For 8-bit quantization

# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set up optimizations for T4 GPU
torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for better performance

# Now implement the application
import os
import gradio as gr
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from yt_dlp import YoutubeDL
from qwen_vl_utils import process_vision_info
from huggingface_hub import notebook_login
import tempfile
from google.colab import userdata

# Print Gradio version for debugging
print(f"Gradio version: {gr.__version__}")

# Create data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Initialize global variables
model = None
processor = None
processed_video_data = None
chat_history = []
video_path = None

def login_to_huggingface(hf_token=None):
    """Login to Huggingface with the provided token or from userdata"""
    if not hf_token:
        try:
            hf_token = userdata.get("HF_TOKEN")
            if not hf_token:
                return "No token provided or found in userdata. Please enter your HF token."
        except:
            return "Failed to get token from userdata. Please enter your HF token manually."

    os.environ["HUGGINGFACE_TOKEN"] = hf_token

    # Use the login function instead of notebook_login with token parameter
    from huggingface_hub import login
    login(token=hf_token)

    return "Successfully logged in to Huggingface!"

def download_youtube_video(youtube_url, progress=gr.Progress()):
    """Download a video from YouTube"""
    progress(0, desc="Starting download...")

    # Create a temporary file with .mp4 extension
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "youtube_video.mp4")

    ydl_opts = {
        "format": "best[height<=720]",  # Limit resolution to 720p for T4 GPU
        "outtmpl": output_path,
        "progress_hooks": [lambda d: progress(d['downloaded_bytes']/d['total_bytes'], desc="Downloading video...")
                          if d['status'] == 'downloading' and d['total_bytes'] > 0 else None],
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        progress(1.0, desc="Download complete!")
        return output_path
    except Exception as e:
        return f"Error downloading video: {str(e)}"

def load_model(hf_token=None, progress=gr.Progress()):
    """Load the Qwen2-VL model and processor"""
    global model, processor

    progress(0.1, desc="Loading model...")
    MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"

#Try to get token from userdata if not provided
    if not hf_token:
        try:
            hf_token = userdata.get("HF_TOKEN")
        except:
            pass

    try:
        # For T4 GPU compatibility, use 8-bit quantization to reduce memory usage
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            token=hf_token,
            torch_dtype=torch.float16,  # Use float16 for T4 GPU
            device_map="auto",
            load_in_8bit=True,  # Use 8-bit quantization for T4 memory constraints
        )
        progress(0.6, desc="Model loaded, loading processor...")

        processor = AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token)
        progress(1.0, desc="Processor loaded!")

        return "Model and processor loaded successfully!"
    except Exception as e:
        return f"Error loading model: {str(e)}"

def process_video(video_file, hf_token=None, progress=gr.Progress()):
    """Process the video for the model"""
    global processed_video_data, video_path, chat_history

    # Reset chat history
    chat_history = []

    if model is None or processor is None:
        load_status = load_model(hf_token, progress)
        if "Error" in load_status:
            return load_status, None

    progress(0.3, desc="Processing video...")
    video_path = video_file

    try:
        # Initial system and video context
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant skilled in analyzing and answering questions about videos. Respond based only on the visual and audio content provided."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 1280 * 720,  # Reduced resolution for T4 compatibility
                        "fps": 0.1  # Lower FPS to reduce memory usage
                    }
                ]
            }
        ]

        progress(0.6, desc="Processing video features...")

        # Removed the max_frames parameter as it's not supported
        image_inputs, video_inputs = process_vision_info(
            messages
        )

        processed_video_data = (image_inputs, video_inputs, messages)
        progress(1.0, desc="Video processed successfully!")

        return "Video processed successfully! You can now chat about the video.", chat_history
    except Exception as e:
        return f"Error processing video: {str(e)}", None

def process_youtube_link(youtube_url, hf_token=None, progress=gr.Progress()):
    """Process a YouTube link to download and analyze the video"""
    progress(0.1, desc="Starting YouTube video processing...")
    video_file = download_youtube_video(youtube_url, progress)

    if isinstance(video_file, str) and "Error" in video_file:
        return video_file, None

    progress(0.5, desc="Video downloaded, processing for analysis...")
    return process_video(video_file, hf_token, progress)

def chat_with_video(user_message, history):
    """Chat with the model about the processed video"""
    global processed_video_data

    if processed_video_data is None:
        return "Please process a video first!", history

    image_inputs, video_inputs, messages = processed_video_data

    # Add new user message to chat
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": user_message}]
    })

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process inputs for the model
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Generating response with memory-efficient settings for T4
    with torch.cuda.amp.autocast(dtype=torch.float16):  # Use mixed precision for T4
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False  # Use greedy decoding to save memory
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]

    # Add assistant response to chat
    messages.append({
        "role": "assistant",
        "content": output_text
    })

    history.append((user_message, output_text))
    return "", history

# Try to get HF token from userdata
default_hf_token = None
try:
    default_hf_token = userdata.get("HF_TOKEN")
    print("HF token found in userdata" if default_hf_token else "No HF token in userdata")
except:
    print("Could not access userdata for HF token")

# Build the Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Video Analysis with Qwen2-VL")
    gr.Markdown("## Running on Google Cloud with T4 GPU")

    with gr.Tab("Setup"):
        with gr.Row():
            with gr.Column():
                hf_token_input = gr.Textbox(
                    label="Hugging Face Token",
                    placeholder="Enter your Hugging Face token here",
                    type="password",
                    value=default_hf_token if default_hf_token else ""
                )
                login_button = gr.Button("Login to Hugging Face")
                login_output = gr.Textbox(label="Login Status")

                login_button.click(
                    fn=login_to_huggingface,
                    inputs=[hf_token_input],
                    outputs=[login_output]
                )

    with gr.Tab("Video Analysis"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Upload YouTube Video")
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="Enter YouTube video URL (e.g., https://www.youtube.com/watch?v=...)"
                )
                process_youtube_button = gr.Button("Process YouTube Video")

                gr.Markdown("## Or Upload Local Video")
                # Using basic Video component without additional parameters
                video_input = gr.Video(label="Upload Video")
                process_video_button = gr.Button("Process Uploaded Video")

                processing_status = gr.Textbox(label="Processing Status")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Chat with Video")
                chatbot = gr.Chatbot(label="Chat History")
                msg = gr.Textbox(label="Your Question", placeholder="Ask me anything about the video...")
                clear_button = gr.Button("Clear Chat")

        # Event handlers
        process_youtube_button.click(
            fn=process_youtube_link,
            inputs=[youtube_url, hf_token_input],
            outputs=[processing_status, chatbot]
        )

        process_video_button.click(
            fn=process_video,
            inputs=[video_input, hf_token_input],
            outputs=[processing_status, chatbot]
        )

        msg.submit(
            fn=chat_with_video,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )

        clear_button.click(
            lambda: ([], []),
            inputs=[],
            outputs=[msg, chatbot]
        )

# Launch the app
app.launch(debug=True)