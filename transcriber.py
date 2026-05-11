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

# Suppress excessive warnings from torchcodec and pyannote
warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torchcodec")

# Configure logging (for errors and debugging)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    # Suppress Hugging Face download progress bars to prevent CLI UI corruption
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    
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

def setup_environment():
    """Load and validate environment variables."""
    load_dotenv()
    
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, "models")
    temp_dir = os.path.join(project_root, "temp")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    os.environ["HF_HOME"] = os.path.join(models_dir, "huggingface")
    os.environ["TORCH_HOME"] = os.path.join(models_dir, "torch")
    os.environ["XDG_CACHE_HOME"] = os.path.join(models_dir, "xdg")
    
    config = {
        "INPUT_DIR": os.getenv("INPUT_DIR", "./input_media"),
        "OUTPUT_DIR": os.getenv("OUTPUT_DIR", "./output_files"),
        "PROMPTS_DIR": os.getenv("PROMPTS_DIR", "./prompts"),
        "TEMP_DIR": temp_dir,
        "FILE_PATTERN": os.getenv("FILE_PATTERN", "*.mp4,*.mp3,*.wav,*.m4a,*.mov"),
        "HUGGINGFACE_TOKEN": os.getenv("HUGGINGFACE_TOKEN"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "SUMMARIZER_PROVIDER": os.getenv("SUMMARIZER_PROVIDER", "openai").lower(),
        "WHISPER_MODEL": os.getenv("WHISPER_MODEL_PATH", "mlx-community/whisper-large-v3-turbo"),
        "DIARIZATION_MODEL": os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    }

    if config["SUMMARIZER_PROVIDER"] == "openai":
        if not config["OPENAI_API_KEY"] or config["OPENAI_API_KEY"] == "your_openai_api_key_here":
            console.print("[bold red]Error:[/bold red] OPENAI_API_KEY not set in .env")
    elif config["SUMMARIZER_PROVIDER"] == "gemini":
        if not config["GEMINI_API_KEY"] or config["GEMINI_API_KEY"] == "your_gemini_api_key_here":
            console.print("[bold red]Error:[/bold red] GEMINI_API_KEY not set in .env")

    if config.get("HUGGINGFACE_TOKEN") and config["HUGGINGFACE_TOKEN"] != "your_huggingface_token_here":
        os.environ["HF_TOKEN"] = config["HUGGINGFACE_TOKEN"]

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

def summarize_transcript(transcript, prompt_file, config):
    """Summarize transcript using the configured provider."""
    provider = config["SUMMARIZER_PROVIDER"]
    
    try:
        with open(prompt_file, "r") as f:
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
        try:
            client = genai.Client(api_key=config["GEMINI_API_KEY"])
            prompt = f"{instructions}\n\nTranscript:\n{transcript}"
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Summary generation failed. Error: {str(e)}"
    
    return "Unsupported summarization provider."

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

def process_file(file_path, prompt_file, config):
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
            result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=config["WHISPER_MODEL"], verbose=True)
        finally:
            sys.stdout = original_stdout
            
        progress.update(task, completed=total_duration, description="[bold green]Transcription complete![/bold green]")
        
    whisper_segments = result["segments"]
    
    times['transcribe'] = time.time() - start_time
    
    # Diarization setup
    diarization_segments = []
    has_diarization = False
    
    console.print("\n[bold cyan]▶ Identifying Speakers (Diarization)...[/bold cyan]")
    try:
        pipeline = Pipeline.from_pretrained(config["DIARIZATION_MODEL"], token=config["HUGGINGFACE_TOKEN"])
        if pipeline is None:
            raise ValueError("Pipeline returned None.")
            
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        pipeline.to(device)
        
        with Status("[bold yellow]Analyzing speaker audio...", console=console) as status:
            diarization_output = pipeline(audio_path)
            
            # Pyannote 4.0 returns DiarizeOutput, fallback to legacy if missing
            if hasattr(diarization_output, 'speaker_diarization'):
                diarization = diarization_output.speaker_diarization
            else:
                diarization = diarization_output
                
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                diarization_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            has_diarization = True
            status.update("[bold green]✓ Speakers identified successfully.")
    except Exception as e:
        error_msg = str(e)
        console.print(Panel(f"[bold red]Speaker Diarization Failed![/bold red]\n\n[yellow]Error:[/yellow] {error_msg}\n\nIf you see a 403 Client Error or Gated repo error, you must accept the terms for ALL underlying models:\n1. Go to https://hf.co/pyannote/speaker-diarization-3.1\n2. Go to https://hf.co/pyannote/segmentation-3.0\n3. Go to https://hf.co/pyannote/speaker-diarization-community-1\n4. Log in and agree to the terms on ALL pages.\n5. Ensure your HUGGINGFACE_TOKEN in .env is correct.\n\n[dim]Continuing transcription WITHOUT speaker labels...[/dim]", border_style="red"))
    
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
        console.print(f"\n[bold cyan]▶ AI Post-processing with {config['SUMMARIZER_PROVIDER'].upper()}...[/bold cyan]")
        with Status(f"[bold yellow]Applying prompt '{prompt_name}'...", console=console) as status:
            start_time = time.time()
            summary = summarize_transcript(full_transcript, prompt_file, config)
            times['post_process'] = time.time() - start_time
            
            summary_path = os.path.join(out_dir, f"{prompt_name}.md")
            with open(summary_path, "w") as f:
                f.write(summary)
            status.update(f"[bold green]✓ AI Output saved to {summary_path}")
            console.print(f"[bold green]✓ Post-processing complete.[/bold green]")
            
    # Cleanup
    if audio_path != file_path and os.path.exists(audio_path):
        os.remove(audio_path)

    show_summary_report(times, Path(file_path).name, out_dir, prompt_name, config['SUMMARIZER_PROVIDER'], config["WHISPER_MODEL"])


def dashboard():
    console.print(Panel("[bold magenta]Transcripta CLI[/bold magenta]\n[dim]Interactive Local Transcription Engine[/dim]", border_style="magenta"))
    config = setup_environment()
    
    while True:
        files = get_files(config)
        prompts = get_prompts(config)
        
        if not files:
            console.print(f"[bold yellow]⚠ No media files found in {config['INPUT_DIR']}. Please add files and try again.[/bold yellow]")
            break
            
        file_choices = [
            ("Transcribe Latest File", files[0]),
        ]
        if Separator:
            file_choices.append(Separator())
        
        for f in files:
            mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(f)))
            file_choices.append((f"{Path(f).name} [dim]({mtime})[/dim]", f))
            
        file_answers = inquirer.prompt([
            inquirer.List('target_file',
                          message="1. Select a file to process",
                          choices=file_choices,
            ),
        ])
        if not file_answers: break
        
        prompt_choices = [
            ("No Post-processing (Transcribe Only)", None),
        ]
        if Separator:
            prompt_choices.append(Separator())
        for p in prompts:
            prompt_choices.append((Path(p).name, p))
            
        prompt_answers = inquirer.prompt([
            inquirer.List('target_prompt',
                          message="2. Select an AI prompt for post-processing",
                          choices=prompt_choices,
            ),
        ])
        if not prompt_answers: break
        
        target_file = file_answers['target_file']
        target_prompt = prompt_answers['target_prompt']
        
        try:
            process_file(target_file, target_prompt, config)
        except Exception as e:
            console.print(f"[bold red]Critical Error during processing:[/bold red] {e}")
            
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
