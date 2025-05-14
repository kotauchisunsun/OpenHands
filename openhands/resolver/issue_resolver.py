import asyncio
import dataclasses
import json
import os
import shutil
from typing import Any
from uuid import uuid4

from termcolor import colored

from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, AppConfig, LLMConfig, SandboxConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.event import Event
from openhands.events.observation import (
    CmdOutputObservation,
)
from openhands.events.stream import EventStreamSubscriber
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)
from openhands.resolver.resolver_output import ResolverOutput
from openhands.resolver.utils import (
    codeact_user_response,
    reset_logger_for_multiprocessing,
)
from openhands.runtime.base import Runtime


class IssueResolver:
    GITLAB_CI = os.getenv('GITLAB_CI') == 'true'

    def __init__(
        self,
        owner: str,
        repo: str,
        platform: ProviderType,
        max_iterations: int,
        output_dir: str,
        llm_config: LLMConfig,
        prompt_template: str,
        issue_type: str,
        repo_instruction: str | None,
        issue_number: int,
        comment_id: int | None,
        sandbox_config: SandboxConfig,
        issue_handler: ServiceContextIssue | ServiceContextPR,
    ) -> None:
        """Initialize the IssueResolver with the given parameters."""
        self.owner = owner
        self.repo = repo
        self.platform = platform
        self.max_iterations = max_iterations
        self.output_dir = output_dir
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.issue_type = issue_type
        self.repo_instruction = repo_instruction
        self.issue_number = issue_number
        self.comment_id = comment_id
        self.sandbox_config = sandbox_config
        self.issue_handler = issue_handler

    def execute_command(
        self, runtime: Runtime, command: str, timeout: int | None = None
    ) -> CmdOutputObservation:
        """Execute a command in the runtime with optional timeout and retries.

        Args:
            runtime: Runtime instance to execute the command
            command: Command to execute
            error_msg: Error message template if command fails
            timeout: Optional timeout in seconds

        Returns:
            CmdOutputObservation from successful command execution

        Raises:
            RuntimeError if command fails
        """
        action = CmdRunAction(command=command)
        if timeout is not None:
            action.set_hard_timeout(timeout)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to {command}. Observation: {obs}')
        return obs

    def initialize_runtime(self, runtime: Runtime) -> None:
        """Initialize the runtime for the agent."""
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Initialization')
        logger.info('-' * 30)

        # Change to workspace directory
        self.execute_command(runtime, 'cd /workspace')

        # Set permissions for GitLab CI
        if self.platform == ProviderType.GITLAB and self.GITLAB_CI:
            self.execute_command(runtime, 'sudo chown -R 1001:0 /workspace/*')

        # Configure git
        self.execute_command(runtime, 'git config --global core.pager ""')

    async def complete_runtime(
        self,
        runtime: Runtime,
        base_commit: str,
    ) -> dict[str, Any]:
        """Complete the runtime for the agent."""
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Completion')
        logger.info('-' * 30)

        # Change to workspace directory
        self.execute_command(runtime, 'cd /workspace')

        # Configure git
        self.execute_command(runtime, 'git config --global core.pager ""')

        self.execute_command(
            runtime, 'git config --global --add safe.directory /workspace'
        )

        # Add all changes
        git_add_cmd = (
            'sudo git add -A'
            if self.platform == ProviderType.GITLAB and self.GITLAB_CI
            else 'git add -A'
        )
        self.execute_command(runtime, git_add_cmd)

        # Get git diff with retries
        n_retries = 0
        git_patch = None
        while n_retries < 5:
            try:
                obs = self.execute_command(
                    runtime,
                    f'git diff --no-color --cached {base_commit}',
                    timeout=600 + 100 * n_retries,
                )
                git_patch = obs.content.strip()
                break
            except RuntimeError:
                n_retries += 1
                logger.info('Failed to get git diff, retrying...')
                await asyncio.sleep(10)
                continue
            except Exception as e:
                raise ValueError(f'Unexpected error type: {type(e)}')

        logger.info('-' * 30)
        logger.info('END Runtime Completion')
        logger.info('-' * 30)
        return {'git_patch': git_patch}

    async def process_issue(
        self,
        issue: Issue,
        base_commit: str,
        issue_handler: ServiceContextIssue | ServiceContextPR,
        reset_logger: bool = False,
    ) -> ResolverOutput:
        # Setup the logger properly
        if reset_logger:
            log_dir = os.path.join(self.output_dir, 'infer_logs')
            reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
        else:
            logger.info(f'Starting fixing issue {issue.number}.')

        workspace_base = os.path.join(
            self.output_dir, 'workspace', f'{issue_handler.issue_type}_{issue.number}'
        )

        # Get the absolute path of the workspace base
        workspace_base = os.path.abspath(workspace_base)
        # write the repo to the workspace
        if os.path.exists(workspace_base):
            shutil.rmtree(workspace_base)
        shutil.copytree(os.path.join(self.output_dir, 'repo'), workspace_base)

        config = AppConfig(
            default_agent='CodeActAgent',
            runtime='docker',
            max_budget_per_task=4,
            max_iterations=self.max_iterations,
            sandbox=self.sandbox_config,
            # do not mount workspace
            workspace_base=workspace_base,
            workspace_mount_path=workspace_base,
            agents={'CodeActAgent': AgentConfig(disabled_microagents=['github'])},
        )
        config.set_llm_config(self.llm_config)

        runtime = create_runtime(config)
        await runtime.connect()

        def on_event(evt: Event) -> None:
            logger.info(evt)

        runtime.event_stream.subscribe(
            EventStreamSubscriber.MAIN, on_event, str(uuid4())
        )

        self.initialize_runtime(runtime)

        instruction, images_urls = issue_handler.get_instruction(
            issue, self.prompt_template, self.repo_instruction
        )
        action = MessageAction(content=instruction, image_urls=images_urls)
        try:
            state: State | None = await run_controller(
                config=config,
                initial_user_action=action,
                runtime=runtime,
                fake_user_response_fn=codeact_user_response,
            )
            if state is None:
                raise RuntimeError('Failed to run the agent.')
        except (ValueError, RuntimeError) as e:
            error_msg = f'Agent failed with error: {str(e)}'
            logger.error(error_msg)
            state = None
            last_error: str | None = error_msg

        # Get git patch
        return_val = await self.complete_runtime(runtime, base_commit)
        git_patch = return_val['git_patch']
        logger.info(
            f'Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------'
        )

        # Serialize histories and set defaults for failed state
        if state is None:
            histories = []
            metrics = None
            success = False
            comment_success = None
            result_explanation = 'Agent failed to run'
            last_error = 'Agent failed to run or crashed'
        else:
            histories = [dataclasses.asdict(event) for event in state.history]
            metrics = state.metrics.get() if state.metrics else None
            success, comment_success, result_explanation = issue_handler.guess_success(
                issue, state.history, git_patch
            )

            if issue_handler.issue_type == 'pr' and comment_success:
                success_log = (
                    'I have updated the PR and resolved some of the issues that were '
                    'cited in the pull request review. Specifically, I identified the '
                    'following revision requests, and all the ones that I think I '
                    'successfully resolved are checked off. All the unchecked ones I '
                    'was not able to resolve, so manual intervention may be required:\n'
                )
                try:
                    explanations = json.loads(result_explanation)
                except json.JSONDecodeError:
                    logger.error(
                        f'Failed to parse result_explanation as JSON: {result_explanation}'
                    )
                    explanations = [str(result_explanation)]

                for success_indicator, explanation in zip(
                    comment_success, explanations
                ):
                    status = (
                        colored('[X]', 'red')
                        if success_indicator
                        else colored('[ ]', 'red')
                    )
                    bullet_point = colored('-', 'yellow')
                    success_log += f'\n{bullet_point} {status}: {explanation}'
                logger.info(success_log)
            last_error = state.last_error if state.last_error else None

        # Save the output
        output = ResolverOutput(
            issue=issue,
            issue_type=issue_handler.issue_type,
            instruction=instruction,
            base_commit=base_commit,
            git_patch=git_patch,
            history=histories,
            metrics=metrics,
            success=success,
            comment_success=comment_success,
            result_explanation=result_explanation,
            error=last_error,
        )
        return output

    async def resolve_issue(
        self,
        reset_logger: bool = False,
    ) -> ResolverOutput | None:
        """Resolve a single issue."""
        # Load dataset
        issues: list[Issue] = self.issue_handler.get_converted_issues(
            issue_numbers=[self.issue_number], comment_id=self.comment_id
        )

        if not issues:
            raise ValueError(
                f'No issues found for issue number {self.issue_number}. Please verify that:\n'
                f'1. The issue/PR #{self.issue_number} exists in the repository {self.owner}/{self.repo}\n'
                f'2. You have the correct permissions to access it\n'
                f'3. The repository name is spelled correctly'
            )

        issue = issues[0]

        if self.comment_id is not None:
            if (
                self.issue_type == 'pr'
                and self.comment_id not in issue.review_comment_ids
            ):
                raise ValueError(
                    f'Comment ID {self.comment_id} not found in PR {self.issue_number}'
                )
            elif (
                self.issue_type == 'issue' and self.comment_id not in issue.comment_ids
            ):
                raise ValueError(
                    f'Comment ID {self.comment_id} not found in issue {self.issue_number}'
                )

        # Process all instances and save all results
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'infer_logs'), exist_ok=True)
        output_file = os.path.join(
            self.output_dir,
            'repo',
            '.git',
            'refs',
            'heads',
            'master',
        )
        if not os.path.exists(output_file):
            raise ValueError(
                f'Could not find {output_file}. Please make sure the repository is cloned correctly.'
            )

        with open(output_file, 'r') as f:
            base_commit = f.read().strip()

        try:
            output = await self.process_issue(
                issue,
                base_commit,
                self.issue_handler,
                reset_logger,
            )
            logger.info(f'Done with issue {issue.number}')
            return output

        except Exception as e:
            logger.error(f'Failed to process issue {issue.number}')
            logger.exception(e)
            raise
