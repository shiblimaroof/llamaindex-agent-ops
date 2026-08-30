import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolverConfig:
    groq_model: str = os.getenv("GROQ_RESOLVER_MODEL", "openai/gpt-oss-120b")
    github_token: str | None = os.getenv("GITHUB_TOKEN")
    repo_owner: str = os.getenv("TARGET_REPO_OWNER", "run-llama")
    repo_name: str = os.getenv("TARGET_REPO_NAME", "llama_index")
    output_path: str = os.getenv("OUTPUT_PATH", "data/raw_issues.jsonl")
    grading_key_path: str = os.getenv("GRADING_KEY_PATH", "data/grading_key.jsonl")
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "data/.collector_checkpoint.json")
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    request_timeout_secs: int = int(os.getenv("REQUEST_TIMEOUT_SECS", "30"))


def load_resolver_config() -> ResolverConfig:
    return ResolverConfig()