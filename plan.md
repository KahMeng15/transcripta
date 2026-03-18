### Part 1: The Master Plan

This is your step-by-step roadmap to getting the application running locally on your Mac.

#### Phase 1: System Preparation
Before running any Python code, your Mac needs the right system-level tools to handle audio and video files.
1. Open your Mac Terminal.
2. Install Homebrew (if you do not have it) by pasting the command from the official Homebrew website.
3. Install the video processing engine by running `brew install ffmpeg`.

#### Phase 2: Credentials Setup
The app relies on two external services for the heavy lifting that happens outside the local transcription.
1. Create a free account on **Hugging Face**. Search for `pyannote/speaker-diarization-3.1`, accept their user conditions, and generate a free Access Token in your account settings.
2. Log into your **OpenAI API** dashboard (or Anthropic/Gemini) and generate an API key for the summarization step.

#### Phase 3: Project Structure
Create a dedicated folder on your Mac for this project. The AI will generate the files, but this is how the architecture will look:

| File/Folder Name | Purpose |
| :--- | :--- |
| `input_media/` | Drop your raw `.mp4`, `.mp3`, or `.wav` files here. |
| `output_files/` | The final transcripts and summaries will be saved here. |
| `.env` | Stores your API keys, Hugging Face token, and folder paths securely. |
| `requirements.txt` | Lists the specific Python packages needed (mlx-whisper, pyannote, etc.). |
| `prompt.txt` | The strict instructions given to the AI on *how* to summarize Manglish/mixed meetings. |
| `transcriber.py` | The main engine that ties everything together. |

#### Phase 4: Execution
You will feed the master prompt (below) into an AI coding assistant like ChatGPT, Claude, or Cursor. It will spit out the exact code for `requirements.txt`, `.env`, `prompt.txt`, and `transcriber.py`. You will save those files, install the requirements via terminal, and run the script.

---

### Part 2: The Ultimate AI Agent Prompt

Copy the entire block below and paste it into your preferred AI coding assistant to generate the complete, production-ready application.

***

**Copy Everything Below This Line:**

> **Role:** You are an elite Python engineer specializing in local machine learning, audio processing, and command-line interfaces.
> 
> **Objective:** Build a production-ready Python CLI application that automates transcription, speaker diarization, and AI summarization. **The application MUST be highly optimized for Apple Silicon (M-Series Macs) and specifically configured to handle Malaysian code-switching (English, Bahasa Malaysia, and Manglish slang).**
> 
> **Core Technologies:**
> * `mlx-whisper` for ultra-fast, local Apple Silicon transcription.
> * `pyannote.audio` for speaker diarization.
> * `ffmpeg` (via standard `subprocess` or `ffmpeg-python`) for extracting audio from video.
> * `openai` for generating the final summary.
> * `python-dotenv` for configuration.
> * `argparse` for the CLI.
> 
> **Strict Requirements & Pipeline:**
> 
> 1. **Configuration (`.env`):** The app must strictly read variables from a `.env` file. Do not hardcode paths or keys. Required variables: `INPUT_DIR`, `OUTPUT_DIR`, `FILE_PATTERN`, `OPENAI_API_KEY`, `HUGGINGFACE_TOKEN`, and `WHISPER_MODEL_PATH` (default this to `mlx-community/whisper-large-v3-turbo` or a suitable mesolitica MLX model for Malaysian context).
> 
> 2. **Apple Silicon Hardware Acceleration:** >     * Transcription: Ensure `mlx-whisper` is utilized properly to leverage the Mac GPU natively.
>     * Diarization: You MUST explicitly set the PyTorch device to `"mps"` (Metal Performance Shaders) when initializing the `pyannote.audio` pipeline.
> 
> 3. **The Processing Loop:**
>     * **Discover:** Scan `INPUT_DIR` for files matching `FILE_PATTERN`.
>     * **Extract:** If the file is a video, silently use `ffmpeg` to extract a 16kHz mono `.wav` file to a temporary location.
>     * **Transcribe:** Pass the audio to `mlx-whisper` to get text segments with timestamps.
>     * **Diarize:** Pass the audio to `pyannote.audio` to get speaker segments with timestamps.
>     * **Align & Merge:** Programmatically align the text segments from Whisper with the speaker segments from Pyannote. The output format must be exact: `[00:00 - 00:15] Speaker 1: Alamak, we need to review the budget.`
>     * **Summarize:** Read the instructions from a local `prompt.txt` file. Send the fully merged transcript + the prompt instructions to the OpenAI API (gpt-4o-mini) to generate a structured meeting summary.
> 
> 4. **Outputs & Cleanup:**
>     * Save two files per input: `[Original_Name]_transcript.txt` and `[Original_Name]_summary.txt` inside `OUTPUT_DIR`.
>     * Automatically delete any temporary `.wav` files created during the extraction phase.
> 
> **Required Deliverables:**
> Provide the following in clearly separated code blocks:
> 1. The complete `requirements.txt` file.
> 2. The `.env` template.
> 3. A sample `prompt.txt` template specifically engineered to instruct the AI to process messy Malaysian code-switching (Manglish/Bahasa/English) and output a clean, professional English summary.
> 4. The highly robust, fully commented `transcriber.py` script. Include error handling for missing files or missing API keys.