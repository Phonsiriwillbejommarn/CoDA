"""
Auto-resume: detect and download the latest checkpoint from HF Hub.
Usage: python cmd/auto_resume.py --repo Phonsiri/CoDA-Gemma2-RED-v1
Prints the local path and step number if a checkpoint is found.
"""
import os
import re
import sys
import argparse

def find_latest_checkpoint(repo_id: str, local_dir: str = "verl_checkpoints/resume"):
    """Check HF Hub for the latest global_step checkpoint and download it."""
    try:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        
        # List all files in the repo
        try:
            files = api.list_repo_files(repo_id, repo_type="model")
        except Exception as e:
            print(f"RESUME_STATUS=no_repo", file=sys.stderr)
            return None, 0
        
        # Find all global_step folders
        step_pattern = re.compile(r"actor/global_step_(\d+)/")
        steps = set()
        for f in files:
            match = step_pattern.search(f)
            if match:
                steps.add(int(match.group(1)))
        
        if not steps:
            print(f"RESUME_STATUS=no_checkpoints", file=sys.stderr)
            return None, 0
        
        latest_step = max(steps)
        subfolder = f"actor/global_step_{latest_step}"
        
        print(f"RESUME_STATUS=found_step_{latest_step}", file=sys.stderr)
        print(f"Downloading checkpoint step {latest_step} from {repo_id}...", file=sys.stderr)
        
        # Download only the latest checkpoint subfolder
        local_path = snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            allow_patterns=f"{subfolder}/*",
        )
        
        checkpoint_path = os.path.join(local_path, subfolder)
        
        # Verify it has model files
        if os.path.exists(checkpoint_path) and any(
            f.endswith(('.safetensors', '.bin', '.pt')) 
            for f in os.listdir(checkpoint_path)
        ):
            print(f"RESUME_STATUS=ready", file=sys.stderr)
            # Print results to stdout for shell to capture
            print(f"RESUME_PATH={checkpoint_path}")
            print(f"RESUME_STEP={latest_step}")
            return checkpoint_path, latest_step
        else:
            print(f"RESUME_STATUS=invalid_checkpoint", file=sys.stderr)
            return None, 0
            
    except ImportError:
        print("RESUME_STATUS=no_huggingface_hub", file=sys.stderr)
        return None, 0
    except Exception as e:
        print(f"RESUME_STATUS=error: {e}", file=sys.stderr)
        return None, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-resume from HF Hub checkpoint")
    parser.add_argument("--repo", type=str, required=True, help="HF Hub repo id (e.g. Phonsiri/CoDA-Gemma2-RED-v1)")
    parser.add_argument("--local_dir", type=str, default="verl_checkpoints/resume", help="Local dir to download to")
    args = parser.parse_args()
    
    path, step = find_latest_checkpoint(args.repo, args.local_dir)
    if path is None:
        sys.exit(1)
