# Video Analysis with Qwen2-VL

This project provides a video analysis tool using the Qwen2-VL model, allowing users to analyze YouTube videos or uploaded videos through a user-friendly interface.

## Features

- Process and analyze YouTube videos or local video uploads
- Interactive chat interface for asking questions about the video content
- GPU-optimized for T4 GPUs with memory-efficient settings
- Support for Hugging Face authentication

## Requirements

- Python 3.8+
- CUDA-capable GPU (optimized for T4)
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-analysis-qwen.git
cd video-analysis-qwen
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face token:
   - Get your token from [Hugging Face](https://huggingface.co/settings/tokens)
   - Add it to the application when prompted

## Usage

1. Run the application:
```bash
python videochat.py
```

2. Open the provided local URL in your browser
3. Enter your Hugging Face token
4. Upload a video or provide a YouTube URL
5. Start chatting about the video content

## Notes

- The application is optimized for T4 GPUs with memory-efficient settings
- Video processing is limited to 720p resolution for optimal performance
- Processing large videos may take some time depending on your hardware

## License

MIT License 