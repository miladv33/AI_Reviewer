# DeepSeek PR Review Action

This GitHub Action automatically reviews pull requests using the DeepSeek AI model via Ollama. It provides detailed feedback on code changes, potential issues, and suggestions for improvement.

## Features

- Automated code review using DeepSeek AI
- Customizable review prompts
- Detailed analysis of PR changes
- Automatic commenting on PRs with review feedback
- Support for different DeepSeek models

## Usage

Add the following workflow to your repository (e.g., `.github/workflows/pr_review.yml`):

```yaml
name: AI PR Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: miladv33/deepseek-pr-review@v1.0.0
        with:
          github-token: ${{ secrets.GITHUB_TOKEN }}
          # Optional: custom prompt
          custom-prompt: "Please review these changes focusing on security and performance."
          # Optional: specify different model
          model: "deepseek-r1:14b"
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `github-token` | GitHub token for commenting on PRs | Yes | `${{ github.token }}` |
| `custom-prompt` | Custom prompt for the AI review | No | "Please review the following pull request changes and provide feedback." |
| `model` | DeepSeek model to use | No | "deepseek-r1:14b" |

## Custom Prompts

There are two ways to customize the review prompt:

1. **Using input parameter:**
```yaml
- uses: your-username/deepseek-pr-review@v1
  with:
    custom-prompt: "Please review focusing on security best practices"
```

2. **Using a file:**
Create a file at `.github/workflows/pr_review_prompt.txt` in your repository. If this file exists, it will be used instead of the input parameter or default prompt.

Note: The file-based approach takes precedence over the input parameter.

## Requirements

- The action requires Ubuntu runners
- Ollama will be automatically installed during execution

## License

MIT License - see LICENSE file

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.