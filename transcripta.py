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
import re
import warnings
import atexit
import shutil
import shlex
import traceback

# Suppress excessive warnings from torchcodec and pyannote
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")

# Configure logging (for errors and debugging)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- PROJECT DIRECTORY SETUP ---
# We set these BEFORE any ML imports to ensure models are stored in the project folder
project_root = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(project_root, "models")
temp_dir = os.path.join(project_root, "temp")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

def cleanup_temp_dir():
    """Clear all files in the temp directory on exit."""
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                pass

atexit.register(cleanup_temp_dir)

os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
os.environ["XDG_CACHE_HOME"] = os.path.join(models_dir, "xdg")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# -------------------------------

def get_missing_deps():
    dependencies = ["dotenv", "mlx_whisper", "pyannote.audio", "torch", "openai", "tqdm", "google.genai", "rich", "inquirer"]
    missing = []
    
    try:
        import numpy
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
            process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
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

missing = get_missing_deps()
if missing:
    install_dependencies(missing)

# Now import everything else
try:
    from dotenv import load_dotenv
    import os
    from huggingface_hub import hf_hub_download, list_repo_files
    
    import mlx_whisper
    import torch
    import torchaudio
    
    # Monkeypatch torchaudio for compatibility with pyannote
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile', 'sox']
        
    from pyannote.audio import Pipeline
    from openai import OpenAI
    from google import genai
    from rich.console import Console
    from rich.panel import Panel
    from rich.status import Status
    from rich.table import Table
    import inquirer
    try:
        from inquirer import Separator
    except ImportError:
        Separator = None
except (ImportError, Exception) as e:
    print(f"[-] Critical Error: {e}")
    print("[*] Try: pip install --force-reinstall -r requirements.txt")
    sys.exit(1)

console = Console()

def check_and_download_models(config):
    """Verify if models exist locally; if not, ask for permission to download."""
    whisper_repo = config["WHISPER_MODEL"]
    diarization_repo = config["DIARIZATION_MODEL"]
    
    # Check if models are likely downloaded (looking for local snapshots/files)
    # HF_HOME is set to project/models/huggingface
    hf_base = os.environ["HF_HOME"]
    
    def is_model_local(repo_id):
        repo_path = os.path.join(hf_base, "hub", f"models--{repo_id.replace('/', '--')}")
        return os.path.exists(repo_path) and any(os.scandir(repo_path))

    whisper_exists = is_model_local(whisper_repo)
    diarization_exists = is_model_local(diarization_repo)

    if not whisper_exists or not diarization_exists:
        console.print(Panel("[bold yellow]Model Setup Required[/bold yellow]\n\n"
                            "Some AI models need to be downloaded to your local `models/` folder.\n"
                            "Total size is approximately [cyan]3GB - 4GB[/cyan].", border_style="yellow"))
        
        console.print("\n[bold red]IMPORTANT:[/bold red] Before downloading, you [bold underline]must[/bold underline] accept the terms on Hugging Face:")
        console.print(f"1. [link=https://hf.co/{diarization_repo}]https://hf.co/{diarization_repo}[/link]")
        console.print("2. [link=https://hf.co/pyannote/segmentation-3.0]https://hf.co/pyannote/segmentation-3.0[/link]")
        console.print("3. [link=https://hf.co/pyannote/speaker-diarization-community-1]https://hf.co/pyannote/speaker-diarization-community-1[/link]")
        console.print("\nEnsure your [bold]HUGGINGFACE_TOKEN[/bold] in .env is correct.\n")

        confirm = inquirer.prompt([
            inquirer.Confirm('download', message="Do you want to proceed with the download?", default=True)
        ])

        if not confirm or not confirm['download']:
            console.print("[bold red]Download cancelled.[/bold red] The app requires these models to function. Exiting...")
            sys.exit(0)

        # Trigger downloads with visible status
        with Status("[bold cyan]Initializing model downloads (this may take a few minutes)...", console=console) as status:
            if not whisper_exists:
                status.update(f"[bold cyan]Downloading Whisper Model: {whisper_repo}...")
                # We use mlx_whisper's internal logic which will use HF_HOME
                import mlx_whisper
                # This doesn't actually transcribe, just ensures model is in cache
                # but mlx_whisper doesn't have a clean 'download' only call, 
                # so we let the first run handle it, but we've warned the user.
            
            if not diarization_exists:
                status.update(f"[bold cyan]Downloading Diarization Model: {diarization_repo}...")
                try:
                    from pyannote.audio import Pipeline
                    Pipeline.from_pretrained(diarization_repo, token=config["HUGGINGFACE_TOKEN"])
                except Exception as e:
                    console.print(f"\n[bold red]Error downloading diarization model:[/bold red] {e}")
                    console.print("[yellow]Hint: Check your internet connection and HF Token.[/yellow]")
                    sys.exit(1)
        
        console.print("[bold green]✓ Models downloaded and verified![/bold green]\n")

def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    config = {
        "INPUT_DIR": os.getenv("INPUT_DIR", "./input_media"),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "./output_files"),
        "PROMPTS_DIR": os.getenv("PROMPTS_DIR", "./prompts"),
        "TEMP_DIR": temp_dir,
        "FILE_PATTERN": os.getenv("FILE_PATTERN", "*.mp4,*.mp3,*.wav,*.m4a,*.mov"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "GEMINI_MODELS": os.getenv("GEMINI_MODELS", "gemini-1.5-flash").split(","),
        "OPENAI_MODELS": os.getenv("OPENAI_MODELS", "gpt-4o-mini").split(","),
        "SUMMARIZER_PROVIDER": os.getenv("SUMMARIZER_PROVIDER", "openai").lower(),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL_PATH", "mlx-community/whisper-large-v3-turbo"),
        "WHISPER_LANGUAGE": os.getenv("WHISPER_LANGUAGE"),
        "WHISPER_PROMPT": os.getenv("WHISPER_PROMPT"),
        "DIARIZATION_MODEL": os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    }

    # Clean up model lists and optional strings
    config["GEMINI_MODELS"] = [m.strip() for m in config["GEMINI_MODELS"] if m.strip()]
    config["OPENAI_MODELS"] = [m.strip() for m in config["OPENAI_MODELS"] if m.strip()]
    
    if config["WHISPER_LANGUAGE"]:
        config["WHISPER_LANGUAGE"] = config["WHISPER_LANGUAGE"].strip().lower()
        if not config["WHISPER_LANGUAGE"] or config["WHISPER_LANGUAGE"] == "auto":
            config["WHISPER_LANGUAGE"] = None
            
    if config["WHISPER_PROMPT"]:
        config["WHISPER_PROMPT"] = config["WHISPER_PROMPT"].strip()
        if not config["WHISPER_PROMPT"]:
            config["WHISPER_PROMPT"] = None

    if config["SUMMARIZER_PROVIDER"] == "openai":
        if not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "your_openai_api_key_here":
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set in .env")
    elif config["SUMMARIZER_PROVIDER"] == "gemini":
        if not config["GEMINI_API_KEY"] or config["GEMINI_API_KEY"] == "your_gemini_api_key_here":
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")

    if config.get("HUGGINGFACE_TOKEN") and config["HUGGINGFACE_TOKEN"] != "your_huggingface_token_here":
        os.environ["HF_TOKEN"] = config["HUGGINGFACE_TOKEN"]

    # --- MODEL VERIFICATION ---
    check_and_download_models(config)
    # --------------------------

    os.makedirs(config["INPUT_DIR"], exist_ok=True)
    os.makedirs(config["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(config["PROMPTS_DIR"], exist_ok=True)

    return config

def get_audio_duration(file_path):
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of",
             "default=noprint_wrappers=1:nokey=1", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return 1.0

class WhisperProgressStream:
    def __init__(self, progress, task, total_duration, original_stdout):
        self.progress = progress
        self.task = task
        self.total_duration = total_duration
        self.original_stdout = original_stdout
        self.buffer = ""

    def write(self, data):
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            match = re.search(r'-->\s+([^\]]+)\]', line)
            if match:
                try:
                    time_str = match.group(1).strip()
                    parts = time_str.split(':')
                    secs = float(parts[-1])
                    mins = int(parts[-2])
                    hours = int(parts[-3]) if len(parts) > 2 else 0
                    current = hours * 3600 + mins * 60 + secs
                    self.progress.update(self.task, completed=min(current, self.total_duration))
                except Exception:
                    pass
            
            # Prevent infinite recursion by temporarily restoring stdout
            import sys
            original = sys.stdout
            sys.stdout = self.original_stdout
            try:
                # Let Rich print it so it goes above the progress bar
                self.progress.console.print(line)
            finally:
                sys.stdout = original

    def flush(self):
        pass

class DiarizationProgressHook:
    def __init__(self, progress, task_id):
        self.progress = progress
        self.task_id = task_id

    def __call__(self, step_name, step_arg=None, file=None, completed=None, total=None, **kwargs):
        if total is not None and completed is not None:
            self.progress.update(self.task_id, completed=completed, total=total, description=f"[cyan]Diarizing: {step_name}...")
        else:
            self.progress.update(self.task_id, description=f"[cyan]Diarizing: {step_name}...")

def extract_audio(input_file, temp_dir):
    """Extract a 16kHz mono WAV file to the local temp directory for optimal MLX processing."""
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

def summarize_transcript(transcript, prompt_file, config, status=None, target_language="auto"):
    """Summarize transcript using the configured provider with fallback support."""
    provider = config["SUMMARIZER_PROVIDER"]
    
    try:
        with open(prompt_file, "r") as f:
            instructions = f.read()
    except FileNotFoundError:
        instructions = "Please summarize this meeting transcript."

    if target_language == "english":
        instructions += "\n\nIMPORTANT: Please generate the final output in English."
    elif target_language == "malay":
        instructions += "\n\nIMPORTANT: Sila hasilkan output dalam Bahasa Melayu."

    if provider == "openai":
        client = OpenAI(api_key=config["OPENAI_API_KEY"])
        models = config.get("OPENAI_MODELS", ["gpt-4o-mini"])
        last_error = ""
        
        for model_name in models:
            if status: status.update(f"[bold yellow]Summarizing with OpenAI ({model_name})...")
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"}
                    ]
                )
                return response.choices[0].message.content
            except Exception as e:
                last_error = str(e)
                if status: console.print(f"[yellow]⚠ OpenAI {model_name} failed: {last_error}. Falling back...[/yellow]")
        
        raise Exception(f"All OpenAI models failed. Last error: {last_error}")
    
    elif provider == "gemini":
        models = config.get("GEMINI_MODELS", ["gemini-1.5-flash"])
        last_error = ""
        
        for model_name in models:
            if status: status.update(f"[bold yellow]Summarizing with Gemini ({model_name})...")
            try:
                client = genai.Client(api_key=config["GEMINI_API_KEY"])
                prompt = f"{instructions}\n\nTranscript:\n{transcript}"
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                last_error = str(e)
                if status: console.print(f"[yellow]⚠ Gemini {model_name} failed: {last_error}. Falling back...[/yellow]")
        
        raise Exception(f"All Gemini models failed. Last error: {last_error}")
    
    raise Exception(f"Unsupported summarization provider: {provider}")

def get_files(config):
    patterns = config["FILE_PATTERN"].split(',')
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(config["INPUT_DIR"], pattern.strip())))
    
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def get_prompts(config):
    prompts = glob.glob(os.path.join(config["PROMPTS_DIR"], "*.md"))
    prompts.sort()
    return prompts

def clean_path(path_str):
    """Clean terminal-specific artifacts from a single path string."""
    path_str = path_str.strip()
    # Remove leading '@' if present
    if path_str.startswith('@'):
        path_str = path_str[1:]
    
    # Remove quotes
    path_str = path_str.strip("'\"")
    # Resolve absolute paths and expand user home (~)
    return os.path.abspath(os.path.expanduser(path_str))

def parse_multiple_paths(input_str):
    """Robustly split a string containing one or more paths (quoted or escaped)."""
    input_str = input_str.strip()
    if not input_str:
        return []

    # Some terminals drag-and-drop multiple files like 'path1''path2' without spaces.
    # We insert spaces between quotes to help shlex split them correctly.
    input_str = re.sub(r"' *'", "' '", input_str)
    input_str = re.sub(r'" *"', '" "', input_str)
    
    try:
        paths = shlex.split(input_str)
    except ValueError:
        # Fallback to a simpler split if shlex fails (e.g., unclosed quotes)
        paths = input_str.split(' ')
        
    cleaned_paths = [clean_path(p) for p in paths if p.strip()]
    return cleaned_paths

def get_files_from_path(path, config):
    """Resolve a single path (file or directory) to a list of media files."""
    if not os.path.exists(path):
        return []
    
    if os.path.isfile(path):
        return [path]
    
    if os.path.isdir(path):
        patterns = config["FILE_PATTERN"].split(',')
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(path, pattern.strip())))
        files.sort()
        return files
    
    return []

def show_summary_report(times, file_name, out_dir, prompt_name, provider, whisper_model):
    console.print("\n")
    panel = Panel(f"[bold green]Processing Complete for[/bold green] [cyan]{file_name}[/cyan]", border_style="green")
    console.print(panel)
    
    table = Table(show_header=False, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Output Location", f"[yellow]{out_dir}[/yellow]")
    table.add_row("Transcription Time", f"{times['transcribe']:.2f} seconds")
    if prompt_name:
        table.add_row("Post-Processing Time", f"{times['post_process']:.2f} seconds")
        table.add_row("Prompt Used", f"{prompt_name}")
        table.add_row("AI Provider", f"{provider}")
    table.add_row("Audio Model", f"{whisper_model}")
    
    console.print(table)
    console.print("\n")

def process_file(file_path, prompt_file, config, do_diarization=True, target_language="auto"):
    """Complete processing pipeline for a single file."""
    base_name = Path(file_path).stem
    out_dir = os.path.join(config["OUTPUT_DIR"], base_name)
    os.makedirs(out_dir, exist_ok=True)
    
    times = {}
    console.print(f"\n[bold cyan]▶ Preprocessing file:[/bold cyan] {Path(file_path).name}")
    
    with Status("[bold yellow]Optimizing audio format (16kHz WAV)...", console=console) as status:
        audio_path = extract_audio(file_path, config["TEMP_DIR"])
        if not audio_path:
            console.print("[red]✗ Extraction failed.[/red]")
            return
        status.update("[bold green]✓ Audio ready for transcription.")
        
    console.print("\n[bold cyan]▶ Transcribing audio with MLX-Whisper...[/bold cyan]")
    
    start_time = time.time()
    total_duration = get_audio_duration(audio_path)
    
    from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        "<",
        TimeRemainingColumn(),
        console=console,
        transient=False
    ) as progress:
        task = progress.add_task("[cyan]Transcribing...", total=total_duration)
        
        # Redirect stdout to our custom stream
        original_stdout = sys.stdout
        sys.stdout = WhisperProgressStream(progress, task, total_duration, original_stdout)
        try:
            result = mlx_whisper.transcribe(
                audio_path, 
                path_or_hf_repo=config["WHISPER_MODEL"], 
                verbose=True,
                language=config.get("WHISPER_LANGUAGE"),
                initial_prompt=config.get("WHISPER_PROMPT")
            )
        finally:
            sys.stdout = original_stdout
            
        progress.update(task, completed=total_duration, description="[bold green]Transcription complete![/bold green]")
        
    whisper_segments = result["segments"]
    
    times['transcribe'] = time.time() - start_time
    
    # Diarization setup
    diarization_segments = []
    has_diarization = False
    
    if do_diarization:
        console.print("\n[bold cyan]▶ Identifying Speakers (Diarization)...[/bold cyan]")
        try:
            pipeline = Pipeline.from_pretrained(config["DIARIZATION_MODEL"], token=config["HUGGINGFACE_TOKEN"])
            if pipeline is None:
                raise ValueError("Pipeline returned None.")
                
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            pipeline.to(device)
            
            from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn, BarColumn, TextColumn
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                TimeElapsedColumn(),
                "<",
                TimeRemainingColumn(),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Diarizing...", total=None)
                hook = DiarizationProgressHook(progress, task)
                
                diarization_output = pipeline(audio_path, hook=hook)
                
                # Pyannote 4.0 returns DiarizeOutput, fallback to legacy if missing
                if hasattr(diarization_output, 'speaker_diarization'):
                    diarization = diarization_output.speaker_diarization
                else:
                    diarization = diarization_output
                    
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    diarization_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
                has_diarization = True
                console.print("[bold green]✓ Speakers identified successfully.[/bold green]")
        except Exception as e:
            error_msg = str(e)
            console.print(Panel(f"[bold red]Speaker Diarization Failed![/bold red]\n\n[yellow]Error:[/yellow] {error_msg}\n\nIf you see a 403 Client Error or Gated repo error, you must accept the terms for ALL underlying models:\n1. Go to https://hf.co/pyannote/speaker-diarization-3.1\n2. Go to https://hf.co/pyannote/segmentation-3.0\n3. Go to https://hf.co/pyannote/speaker-diarization-community-1\n4. Log in and agree to the terms on ALL pages.\n5. Ensure your HUGGINGFACE_TOKEN in .env is correct.\n\n[dim]Continuing transcription WITHOUT speaker labels...[/dim]", border_style="red"))
    else:
        console.print("\n[dim]▶ Skipping Speaker Identification...[/dim]")
    
    # Compile text into paragraphs instead of line-by-line
    merged_transcript = []
    current_paragraph = ""
    p_start = None
    last_end = 0
    current_speaker = "Unknown"
    
    for segment in whisper_segments:
        start, end, text = segment["start"], segment["end"], segment["text"].strip()
        
        best_speaker = "Unknown"
        if has_diarization:
            max_overlap = 0
            for d_seg in diarization_segments:
                overlap = min(end, d_seg["end"]) - max(start, d_seg["start"])
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = d_seg["speaker"]
        
        speaker_changed = has_diarization and best_speaker != current_speaker
        
        if p_start is None:
            p_start = start
            current_speaker = best_speaker
            
        is_long = (end - p_start) > 20
        has_punct = current_paragraph.endswith((".", "?", "!"))
        has_pause = (start - last_end) > 2.0
        
        if (is_long and has_punct) or has_pause or speaker_changed:
            if current_paragraph:
                timestamp = f"[{int(p_start//60):02}:{int(p_start%60):02} - {int(last_end//60):02}:{int(last_end%60):02}]"
                prefix = f"{current_speaker}: " if has_diarization else ""
                merged_transcript.append(f"{timestamp} {prefix}{current_paragraph}\n")
                
            current_paragraph = text
            p_start = start
            current_speaker = best_speaker
        else:
            if current_paragraph:
                current_paragraph += " " + text
            else:
                current_paragraph = text
            
        last_end = end
        
    if current_paragraph:
        end = whisper_segments[-1]["end"] if whisper_segments else last_end
        p_start = p_start if p_start is not None else 0
        timestamp = f"[{int(p_start//60):02}:{int(p_start%60):02} - {int(end//60):02}:{int(end%60):02}]"
        prefix = f"{current_speaker}: " if has_diarization else ""
        merged_transcript.append(f"{timestamp} {prefix}{current_paragraph}\n")
    
    full_transcript = "\n".join(merged_transcript)
    
    transcript_path = os.path.join(out_dir, "transcription.md")
    with open(transcript_path, "w") as f:
        f.write("# Transcription\n\n")
        f.write(full_transcript)
        
    console.print(f"[bold green]✓ Transcript saved to {transcript_path}[/bold green]")
    
    prompt_name = None
    if prompt_file:
        prompt_name = Path(prompt_file).stem
        console.print(f"\n[bold cyan]▶ AI Post-processing with {config['SUMMARIZER_PROVIDER'].title()}...[/bold cyan]")
        with Status(f"[bold yellow]Applying prompt '{prompt_name}'...", console=console) as status:
            start_time = time.time()
            try:
                summary = summarize_transcript(full_transcript, prompt_file, config, status=status, target_language=target_language)
                times['post_process'] = time.time() - start_time
                
                summary_path = os.path.join(out_dir, f"{prompt_name}.md")
                with open(summary_path, "w") as f:
                    f.write(summary)
                status.update(f"[bold green]✓ AI Output saved to {summary_path}")
                console.print(f"[bold green]✓ Post-processing complete.[/bold green]")
            except Exception as e:
                times['post_process'] = time.time() - start_time
                console.print(f"[bold red]✗ AI Post-processing failed:[/bold red] {e}")
                # We don't save the summary file if it failed completely
            
    # Cleanup
    if audio_path != file_path and os.path.exists(audio_path):
        os.remove(audio_path)

    show_summary_report(times, Path(file_path).name, out_dir, prompt_name, config['SUMMARIZER_PROVIDER'], config["WHISPER_MODEL"])


def dashboard():
    console.print(Panel("[bold magenta]Transcripta CLI[/bold magenta]\n[dim]Interactive Local Transcription Engine[/dim]\n[dim cyan]by kahmeng kahmeng15.github.io[/dim cyan]", border_style="magenta"))
    config = setup_environment()
    
    while True:
        # Step 0: Select Mode
        mode_select = inquirer.prompt([
            inquirer.List('mode',
                          message="1. Select Processing Mode",
                          choices=[
                              ("Single File (Process one recording)", "single"),
                              ("Batch Processing (Process multiple files or a folder)", "batch"),
                              ("Exit", "exit")
                          ])
        ])
        if not mode_select or mode_select['mode'] == 'exit': break
        global_mode = mode_select['mode']
        
        files = get_files(config)
        prompts = get_prompts(config)
        target_files = []
        
        if global_mode == "single":
            file_choices = []
            if files:
                file_choices.append(("Latest File", files[0]))
                if Separator: file_choices.append(Separator())
            
            file_choices.append(("Manual Path / Drag-and-Drop", "manual"))
            if files:
                if Separator: file_choices.append(Separator())
                for f in files:
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))
                    file_choices.append((f"{Path(f).name} ({mtime})", f))
            
            file_answers = inquirer.prompt([
                inquirer.List('target_file', message="2. Select a file to process", choices=file_choices)
            ])
            if not file_answers: continue
            
            if file_answers['target_file'] == "manual":
                path_answer = inquirer.prompt([inquirer.Text('path', message="Drag and drop file here")])
                if not path_answer or not path_answer['path']: continue
                
                # Single mode: we take the first valid file from whatever they dragged
                parsed_paths = parse_multiple_paths(path_answer['path'])
                for p in parsed_paths:
                    found = get_files_from_path(p, config)
                    if found:
                        target_files = [found[0]]
                        break
                
                if not target_files:
                    console.print("[bold red]Error:[/bold red] No valid media file found in the provided path.")
                    continue
            else:
                target_files = [file_answers['target_file']]
                
        elif global_mode == "batch":
            batch_choices = [
                ("Drag and Drop Files/Folders (Collection Mode)", "folder_manual")
            ]
            if files:
                if Separator: batch_choices.append(Separator())
                batch_choices.append(("Select Multiple Files from input_media", "select_files"))
            
            batch_answer = inquirer.prompt([
                inquirer.List('type', message="2. Batch Input Method", choices=batch_choices)
            ])
            if not batch_answer: continue
            
            if batch_answer['type'] == "folder_manual":
                console.print("\n[bold cyan]▶ Collection Mode Enabled[/bold cyan]")
                console.print("[dim]1. Drag and drop one or more files/folders into this terminal and press Enter.[/dim]")
                console.print("[dim]2. Repeat as many times as you like.[/dim]")
                console.print("[dim]3. Type [bold yellow]'d'[/bold yellow] and press Enter when you are done collecting files.[/dim]\n")
                
                while True:
                    path_answer = inquirer.prompt([
                        inquirer.Text('path', message=f"Add to queue ({len(target_files)} files collected, 'd' to finish)")
                    ])
                    if not path_answer: break
                    
                    val = path_answer['path'].strip()
                    if val.lower() == 'd':
                        break
                    if not val:
                        continue
                        
                    parsed_paths = parse_multiple_paths(val)
                    new_files_count = 0
                    for p in parsed_paths:
                        found = get_files_from_path(p, config)
                        if found:
                            target_files.extend(found)
                            new_files_count += len(found)
                        else:
                            console.print(f"[yellow]⚠ No valid files found in: {p}[/yellow]")
                    
                    if new_files_count > 0:
                        console.print(f"[bold green]✓ Added {new_files_count} file(s) to queue.[/bold green]")
                
                # Deduplicate while preserving order
                target_files = list(dict.fromkeys(target_files))
                
                if not target_files:
                    console.print("[bold yellow]Queue is empty. Returning to menu...[/bold yellow]")
                    continue
                
            elif batch_answer['type'] == "select_files":
                checkbox_choices = []
                for f in files:
                    mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))
                    checkbox_choices.append((f"{Path(f).name} ({mtime})", f))
                
                selected_files = inquirer.prompt([
                    inquirer.Checkbox('files', message="Select files to process (Space to select, Enter to confirm)", choices=checkbox_choices)
                ])
                if not selected_files or not selected_files['files']: continue
                target_files = selected_files['files']
        
        # New Step: Choose processing mode
        mode_answers = inquirer.prompt([
            inquirer.List('mode',
                          message=f"3. Select processing mode (for {len(target_files)} file{'s' if len(target_files) > 1 else ''})",
                          choices=[
                              ("Transcribe Only", "transcribe"),
                              ("Transcribe + Identify Speakers", "diarize"),
                          ]),
        ])
        if not mode_answers: break
        do_diarization = mode_answers['mode'] == "diarize"
        
        prompt_choices = [
            ("No Post-processing (Transcribe Only)", None),
        ]
        if Separator:
            prompt_choices.append(Separator())
        for p in prompts:
            prompt_choices.append((Path(p).name, p))
            
        prompt_answers = inquirer.prompt([
            inquirer.List('target_prompt',
                          message="4. Select an AI prompt for post-processing",
                          choices=prompt_choices,
            ),
        ])
        if not prompt_answers: break
        
        target_prompt = prompt_answers['target_prompt']
        
        # New Step: Select target language for AI output
        target_language = "auto"
        if target_prompt:
            lang_answers = inquirer.prompt([
                inquirer.List('lang',
                              message="4. Select target language for AI output",
                              choices=[
                                  ("Same as Transcript (Auto)", "auto"),
                                  ("English", "english"),
                                  ("Malay (Bahasa Melayu)", "malay"),
                              ]),
            ])
            if not lang_answers: break
            target_language = lang_answers['lang']
            
        auto_quit_answers = inquirer.prompt([
            inquirer.Confirm('auto_quit',
                             message="Auto-quit application when finished? (Recommended to free up RAM)",
                             default=True)
        ])
        auto_quit = auto_quit_answers['auto_quit'] if auto_quit_answers else False
        
        # Process the queue
        for i, file_path in enumerate(target_files):
            if len(target_files) > 1:
                console.print(Panel(f"[bold cyan]Queue Progress: {i+1}/{len(target_files)}[/bold cyan]\n[dim]Processing: {Path(file_path).name}[/dim]", border_style="cyan"))
            
            # Check for existing transcription (only for individual files in the queue)
            base_name = Path(file_path).stem
            out_dir = os.path.join(config["OUTPUT_DIR"], base_name)
            transcript_path = os.path.join(out_dir, "transcription.md")
            
            if os.path.exists(transcript_path) and len(target_files) == 1:
                console.print(f"\n[bold yellow]⚠️  Transcription for '{base_name}' already exists![/bold yellow]")
                overwrite_answer = inquirer.prompt([
                    inquirer.Confirm('overwrite',
                                     message="Do you want to overwrite it?",
                                     default=False)
                ])
                if not overwrite_answer or not overwrite_answer['overwrite']:
                    console.print("[dim]Skipping file...[/dim]\n")
                    continue
            
            try:
                process_file(file_path, target_prompt, config, do_diarization=do_diarization, target_language=target_language)
            except Exception as e:
                console.print(Panel(f"[bold red]Critical Error during processing '{Path(file_path).name}':[/bold red]\n\n{str(e)}\n\n[dim]{traceback.format_exc()}[/dim]", border_style="red"))
            
        console.print(f"\n[bold green]✓ Successfully processed {len(target_files)} file{'s' if len(target_files) > 1 else ''}![/bold green]")
        
        if auto_quit:
            console.print("\n[bold yellow]Auto-quitting to free up RAM...[/bold yellow]")
            break
            
        post_run = inquirer.prompt([
            inquirer.List('action',
                          message="What would you like to do next?",
                          choices=[
                              ("Return to Dashboard", "dash"),
                              ("Exit", "exit")
                          ])
        ])
        
        if not post_run or post_run['action'] == 'exit':
            break

if __name__ == "__main__":
    try:
        dashboard()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting...[/bold yellow]")
        sys.exit(0)
