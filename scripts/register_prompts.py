#!/usr/bin/env python3
"""Register LLM prompts in MLflow Prompt Registry.

Versions prompts alongside models. Each prompt change creates a new version
with a commit message. Supports aliases (staging, production) for deployment.

Usage:
    python scripts/register_prompts.py                    # register/update all prompts
    python scripts/register_prompts.py --alias production  # set alias on latest
    python scripts/register_prompts.py --list              # list all prompt versions
"""

import argparse
import logging
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.observability import setup_logging

setup_logging()
logger = logging.getLogger("hazmat.prompts")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DB = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

# Prompt names in the registry
SYSTEM_PROMPT_NAME = "hazmat-system-prompt"
USER_PROMPT_NAME = "hazmat-user-prompt"


def get_system_prompt_template() -> str:
    """Get the current system prompt (with MLflow template variables)."""
    from src.llm_fallback import SYSTEM_PROMPT
    return SYSTEM_PROMPT


def get_user_prompt_template() -> str:
    """User prompt template with variables for item data."""
    return (
        "Classify the following products:\n\n"
        "{{ items }}"
    )


def register_prompts(commit_message: str = "Update prompts"):
    """Register or update prompts in MLflow Prompt Registry."""
    mlflow.set_tracking_uri(MLFLOW_DB)

    # Register system prompt
    system_template = get_system_prompt_template()
    try:
        prompt = mlflow.genai.register_prompt(
            name=SYSTEM_PROMPT_NAME,
            template=system_template,
            commit_message=commit_message,
            tags={
                "type": "system",
                "model": "gemini-flash-latest",
                "domain": "hazmat-classification",
                "language": "pt-BR",
                "temperature": "0",
                "response_format": "application/json",
            },
        )
        logger.info("Registered system prompt: %s (version %d)", SYSTEM_PROMPT_NAME, prompt.version)
    except Exception as e:
        logger.error("Failed to register system prompt: %s", e)
        return

    # Register user prompt template
    user_template = get_user_prompt_template()
    try:
        prompt = mlflow.genai.register_prompt(
            name=USER_PROMPT_NAME,
            template=user_template,
            commit_message=commit_message,
            tags={
                "type": "user",
                "items_per_request": "20",
                "max_description_length": "300",
            },
        )
        logger.info("Registered user prompt: %s (version %d)", USER_PROMPT_NAME, prompt.version)
    except Exception as e:
        logger.error("Failed to register user prompt: %s", e)


def set_alias(alias: str):
    """Set alias (staging/production) on latest prompt versions."""
    mlflow.set_tracking_uri(MLFLOW_DB)
    client = MlflowClient()

    for name in [SYSTEM_PROMPT_NAME, USER_PROMPT_NAME]:
        try:
            # Get latest version
            prompt = mlflow.genai.load_prompt(name)
            client.set_registered_model_alias(name, alias, str(prompt.version))
            logger.info("Set alias '%s' on %s version %d", alias, name, prompt.version)
        except Exception as e:
            logger.warning("Failed to set alias on %s: %s", name, e)


def list_prompts():
    """List all registered prompt versions."""
    mlflow.set_tracking_uri(MLFLOW_DB)

    logger.info("=" * 70)
    logger.info("PROMPT REGISTRY")
    logger.info("=" * 70)

    for name in [SYSTEM_PROMPT_NAME, USER_PROMPT_NAME]:
        try:
            prompt = mlflow.genai.load_prompt(name)
            logger.info("")
            logger.info("  %s (latest: v%d)", name, prompt.version)
            # Show first 100 chars of template
            preview = prompt.template[:100].replace("\n", " ")
            logger.info("    Preview: %s...", preview)
        except Exception as e:
            logger.info("  %s: not registered (%s)", name, e)

    logger.info("")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", "-m", type=str, default="Update prompts",
                        help="Commit message for prompt version")
    parser.add_argument("--alias", type=str, choices=["staging", "production"],
                        help="Set alias on latest versions")
    parser.add_argument("--list", action="store_true", help="List prompt versions")
    args = parser.parse_args()

    if args.list:
        list_prompts()
        return

    register_prompts(commit_message=args.message)

    if args.alias:
        set_alias(args.alias)


if __name__ == "__main__":
    main()
