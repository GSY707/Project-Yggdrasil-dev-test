# Project Yggdrasil

English (current) | [中文](README.md)

Deterministic memory tree engine for LLM agents.

## Repository Contents

- `src/`: core Yggdrasil memory engine and web server
- `portable_llm/`: portable multi-provider LLM client package
- `prompts/`: system and context-compression prompts
- `特性/`: architecture and design documents

## Quick Start

1. Install dependencies for the engine in `src/`.
2. Start the async web server:

   `cd src && python -m yggdrasil.async_web`

3. Open `http://localhost:8000`.

Detailed engine usage is documented in [src/README.md](src/README.md).

## License

This project is licensed under GNU Affero General Public License v3.0 (AGPL-3.0).

See [LICENSE](LICENSE).

## Security

Please report vulnerabilities privately. See [SECURITY.md](SECURITY.md).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening pull requests.
