import argparse
import asyncio

from openhands.resolver.issue_resolver import IssueResolver


def int_or_none(value: str) -> int | None:
    """Convert string to int or None."""
    return int(value) if value is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--selected-repo',
        required=True,
        help='Repository to analyze in owner/repo format',
    )
    parser.add_argument('--token', help='Token to access the repository')
    parser.add_argument('--username', help='Username to access the repository')
    parser.add_argument(
        '--base-container-image',
        help='Base container image to use (experimental)',
    )
    parser.add_argument(
        '--runtime-container-image', help='Runtime container image to use'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=10,
        help='Maximum number of iterations to run',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory to write the results',
    )
    parser.add_argument(
        '--prompt-file',
        type=str,
        help='Prompt template file to use',
    )
    parser.add_argument(
        '--repo-instruction-file',
        type=str,
        help='Repository instruction file to use',
    )
    parser.add_argument(
        '--issue-type',
        type=str,
        default='issue',
        choices=['issue', 'pr'],
        help='Type of issue to resolve',
    )
    parser.add_argument(
        '--issue-number',
        type=int,
        help='Issue number to resolve',
    )
    parser.add_argument(
        '--comment-id',
        type=int,
        help='Optional ID of a specific comment to focus on',
    )
    parser.add_argument(
        '--base-domain',
        type=str,
        help='Base domain for the git server',
    )
    parser.add_argument(
        '--is-experimental',
        action='store_true',
        help='Enable experimental features',
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        help='Model to use for inference',
    )
    parser.add_argument(
        '--llm-api-key',
        type=str,
        help='API key for the language model',
    )
    parser.add_argument(
        '--llm-base-url',
        type=str,
        help='Base URL for the language model',
    )

    args = parser.parse_args()

    resolver = IssueResolver(args)
    asyncio.run(resolver.resolve_issue())


if __name__ == '__main__':
    main()
