"""Minimal usage example for portable_llm."""

from pathlib import Path

from portable_llm import UnifiedLLMClient


def main() -> None:
    config = {
        "vertex": {
            "project_id": "your-gcp-project-id",
            "service_account_email": "your-service-account@your-project.iam.gserviceaccount.com",
        }
    }

    client = UnifiedLLMClient(
        model_name="gemini-2.0-flash",
        config=config,
        keys_path=str(Path(__file__).with_name("keys.yaml")),
    )

    response = client.generate(
        system_prompt="You are a helpful assistant.",
        user_prompt="Reply with only OK.",
        temperature=0.0,
    )

    print(response.text)
    print(response.usage)


if __name__ == "__main__":
    main()