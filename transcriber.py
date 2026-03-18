#!/usr/bin/env python3
import os
import sys
import glob
import subprocess
import argparse
import logging
from pathlib import Path
import tempfile
import time
import importlib.util

# Configure logging (for errors and debugging)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_missing_deps():
    dependencies = ["dotenv", "mlx_whisper", "pyannote.audio", "torch", "openai", "tqdm", "google.generativeai", "rich", "inquirer"]
    missing = []
    
    # Check NumPy version
    try:
        import numpy
        if int(numpy.__version__.split('.')[0]) >= 2:
            missing.append("numpy<2 (downgrade required)")
    except ImportError:
        missing.append("numpy")

    for dep in dependencies:
        try:
            if importlib.util.find_spec(dep.split('.')[0]) is None:
                missing.append(dep)
        except (ModuleNotFoundError, AttributeError):
            missing.append(dep)
    return missing

def install_dependencies(missing):
    # Try to import rich for a better UI, if not available use simple prints
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.live import Live
        console = Console()
    except ImportError:
        # If rich is missing, we must install it first to show the nice UI
        print(f"[*] Preparing environment (installing UI components)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "rich", "inquirer"])
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.live import Live
        console = Console()

    console.print(Panel("[bold cyan]Transcripta - Environment Setup[/bold cyan]\n[dim]Optimizing your Mac for local AI transcription...[/dim]", border_style="cyan"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(description="Installing missing libraries...", total=None)
        try:
            # Run the actual pip install
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Show the last line of pip output in the progress description
            for line in process.stdout:
                if line.strip():
                    short_line = line.strip()[:60] + "..." if len(line.strip()) > 60 else line.strip()
                    progress.update(task, description=f"[bold yellow]Installing:[/bold yellow] [dim]{short_line}[/dim]")
            
            process.wait()
            if process.returncode != 0:
                raise Exception("Pip install failed")
                
            progress.update(task, description="[bold green]✓ Environment ready![/bold green]")
            time.sleep(1)
        except Exception as e:
            console.print(f"[bold red]Error during setup:[/bold red] {e}")
            sys.exit(1)

    console.print("[bold green]Setup Complete![/bold green] Please restart the script to apply changes.\n")
    sys.exit(0)

# Check and install if needed
missing = get_missing_deps()
if missing:
    install_dependencies(missing)

# Now import everything else
try:
    import torch
    from dotenv import load_dotenv
    import mlx_whisper
    from pyannote.audio import Pipeline
    from openai import OpenAI
    import google.generativeai as genai
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    import inquirer
    try:
        from inquirer import Separator
    except ImportError:
        Separator = None
except (ImportError, Exception) as e:
    # If we still fail, it's likely a complex circular import or env issue
    print(f"[-] Critical Error: {e}")
    print("[*] Try: pip install --force-reinstall -r requirements.txt")
    sys.exit(1)

console = Console()

def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    # Force models and cache to stay within the project directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    temp_dir = os.path.join(project_root, "temp")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set environment variables for AI libraries to use local paths
    os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
    os.environ["XDG_CACHE_HOME"] = os.path.join(models_dir, "xdg")
    
    config = {
        "INPUT_DIR": os.getenv("INPUT_DIR", "./input_media"),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "./output_files"),
        "TEMP_DIR": temp_dir,
        "FILE_PATTERN": os.getenv("FILE_PATTERN", "*.mp4,*.mp3,*.wav"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "SUMMARIZER_PROVIDER": os.getenv("SUMMARIZER_PROVIDER", "openai").lower(),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL_PATH", "mlx-community/whisper-large-v3-turbo")
    }

    if not config["HUGGINGFACE_TOKEN"] or config["HUGGINGFACE_TOKEN"] == "your_huggingface_token_here":
        console.print("[bold red]Error:[/bold red] HUGGINGFACE_TOKEN not set in .env")
        sys.exit(1)
    
    # Provider-specific validation
    if config["SUMMARIZER_PROVIDER"] == "openai":
        if not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "your_openai_api_key_here":
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set in .env")
            sys.exit(1)
    elif config["SUMMARIZER_PROVIDER"] == "gemini":
        if not config["GEMINI_API_KEY"] or config["GEMINI_API_KEY"] == "your_gemini_api_key_here":
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")
            sys.exit(1)

    os.makedirs(config["INPUT_DIR"], exist_ok=True)
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)

    return config

def extract_audio(input_file, temp_dir):
    """Extract a 16kHz mono .wav file to the local temp directory."""
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False).name
    
    try:
        command = [
            "ffmpeg", "-y", "-i", str(input_file),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            temp_wav
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_wav
    except Exception as e:
        if os.path.exists(temp_wav): os.remove(temp_wav)
        return None

def summarize_transcript(transcript, config):
    """Summarize transcript using the configured provider (OpenAI or Gemini)."""
    provider = config["SUMMARIZER_PROVIDER"]
    
    try:
        with open("prompt.txt", "r") as f:
            instructions = f.read()
    except FileNotFoundError:
        instructions = "Please summarize this meeting transcript."

    if provider == "openai":
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed. Error: {str(e)}"
    
    elif provider == "gemini":
        genai.configure(api_key=config["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-1.5-flash")
        try:
            prompt = f"{instructions}\n\nTranscript:\n{transcript}"
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Summary generation failed. Error: {str(e)}"
    
    return "Unsupported summarization provider."

def process_file(file_path, config):
    """Complete processing pipeline for a single file with Rich UI."""
    file_name = Path(file_path).name
    console.print(Panel(f"[bold cyan]Processing:[/bold cyan] {file_name}", expand=False))
    
    with Status("[bold yellow]Extracting audio...", console=console) as status:
        audio_path = extract_audio(file_path, config["TEMP_DIR"])
        if not audio_path:
            console.print("[red]✗ Extraction failed.[/red]")
            return
        status.update("[bold green]✓ Audio extracted.")
        
        # Transcription
        status.update("[bold yellow]Transcribing with MLX-Whisper...")
        whisper_segments = mlx_whisper.transcribe(audio_path, path_or_hf_repo=config["WHISPER_MODEL"], verbose=False)["segments"]
        status.update("[bold green]✓ Transcription complete.")
        
        # Diarization
        status.update("[bold yellow]Identifying speakers (Diarization)...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=config["HUGGINGFACE_TOKEN"])
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        diarization = pipeline(audio_path)
        
        diarization_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
        status.update("[bold green]✓ Diarization complete.")
        
        # Merge
        status.update("[bold yellow]Aligning and merging data...")
        merged_transcript = []
        for segment in whisper_segments:
            start, end, text = segment["start"], segment["end"], segment["text"].strip()
            best_speaker = "Unknown"
            max_overlap = 0
            for d_seg in diarization_segments:
                overlap = min(end, d_seg["end"]) - max(start, d_seg["start"])
                if overlap > max_overlap:
                    max_overlap, best_speaker = overlap, d_seg["speaker"]
            timestamp = f"[{int(start//60):02}:{int(start%60):02} - {int(end//60):02}:{int(end%60):02}]"
            merged_transcript.append(f"{timestamp} {best_speaker}: {text}")
        
        full_transcript = "\n".join(merged_transcript)
        base_name = Path(file_path).stem
        transcript_path = os.path.join(config["OUTPUT_DIR"], f"{base_name}_transcript.txt")
        with open(transcript_path, "w") as f:
            f.write(full_transcript)
        status.update(f"[bold green]✓ Transcript saved to {transcript_path}")
        
        # Summarization
        status.update(f"[bold yellow]Summarizing via {config['SUMMARIZER_PROVIDER'].upper()}...")
        summary = summarize_transcript(full_transcript, config)
        summary_path = os.path.join(config["OUTPUT_DIR"], f"{base_name}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)
        status.update(f"[bold green]✓ Summary saved to {summary_path}")
        
        # Cleanup
        if os.path.exists(audio_path):
            os.remove(audio_path)

    console.print(f"[bold green]Done![/bold green] Results for [cyan]{file_name}[/cyan] are ready.\n")

def get_files(config):
    patterns = config["FILE_PATTERN"].split(',')
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(config["INPUT_DIR"], pattern.strip())))
    
    # Sort by modification time (most recent first)
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def main():
    console.print(Panel("[bold magenta]Transcripta[/bold magenta]\n[dim]Ultra-Fast Local Transcription & Speaker ID[/dim]", border_style="magenta"))

    parser = argparse.ArgumentParser(description="Transcripta: MLX-Powered Transcription & Diarization")
    parser.add_argument("--input", help="Override input directory")
    parser.add_argument("--recent", action="store_true", help="Process only the most recent file")
    parser.add_argument("--all", action="store_true", help="Process all files in the directory")
    args = parser.parse_args()

    config = setup_environment()
    if args.input: config["INPUT_DIR"] = args.input

    files = get_files(config)
    if not files:
        console.print(f"[bold yellow]⚠ No files found in {config['INPUT_DIR']}[/bold yellow]")
        return

    files_to_process = []

    if args.recent:
        files_to_process = [files[0]]
    elif args.all:
        files_to_process = files
    else:
        # Interactive Selection
        choices = [
            ("Process MOST RECENT file", "recent"),
            ("Process ALL files", "all"),
        ]
        
        if Separator:
            choices.append(Separator())
        
        for f in files:
            mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(f)))
            choices.append((f"{Path(f).name} [dim]({mtime})[/dim]", f))

        questions = [
            inquirer.List('choice',
                          message="What would you like to process?",
                          choices=choices,
            ),
        ]
        
        answers = inquirer.prompt(questions)
        if not answers: return
        
        choice = answers['choice']
        if choice == "recent":
            files_to_process = [files[0]]
        elif choice == "all":
            files_to_process = files
        else:
            files_to_process = [choice]

    for file_path in files_to_process:
        try:
            process_file(file_path, config)
        except Exception as e:
            console.print(f"[bold red]Error processing {file_path}:[/bold red] {e}")

if __name__ == "__main__":
    main()
