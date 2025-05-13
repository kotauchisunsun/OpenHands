import asyncio
import dataclasses
import json
import os
import shutil
from argparse import Namespace
from typing import Any
from uuid import uuid4

from pydantic import SecretStr
from termcolor import colored

import openhands
from openhands.controller.state.state import State
from openhands.core.config import AgentConfig, AppConfig, LLMConfig, SandboxConfig
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import create_runtime, run_controller
from openhands.events.action import CmdRunAction, MessageAction
from openhands.events.event import Event
from openhands.events.observation import (
    CmdOutputObservation,
    ErrorObservation,
    Observation,
)
from openhands.events.stream import EventStreamSubscriber
from openhands.integrations.service_types import ProviderType
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import (
    ServiceContextIssue,
    ServiceContextPR,
)
from openhands.resolver.issue_handler_factory import IssueHandlerFactory
from openhands.resolver.resolver_output import ResolverOutput
from openhands.resolver.utils import (
    codeact_user_response,
    get_unique_uid,
    identify_token,
    reset_logger_for_multiprocessing,
)
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import GENERAL_TIMEOUT, call_async_from_sync


class IssueResolver:
    GITLAB_CI = os.getenv('GITLAB_CI') == 'true'

    def __init__(self, args: Namespace) -> None:
        """Initialize the IssueResolver with the given parameters."""
        # Setup and validate container images
        self.sandbox_config = self._setup_sandbox_config(
            args.base_container_image,
            args.runtime_container_image,
            args.is_experimental,
        )

        parts = args.selected_repo.rsplit('/', 1)
        if len(parts) < 2:
            raise ValueError('Invalid repository format. Expected owner/repo')
        owner, repo = parts

        token = args.token or os.getenv('GITHUB_TOKEN') or os.getenv('GITLAB_TOKEN')
        username = args.username if args.username else os.getenv('GIT_USERNAME')
        if not username:
            raise ValueError('Username is required.')

        if not token:
            raise ValueError('Token is required.')

        platform = call_async_from_sync(
            identify_token,
            GENERAL_TIMEOUT,
            token,
            args.base_domain,
        )

        api_key = args.llm_api_key or os.environ['LLM_API_KEY']
        model = args.llm_model or os.environ['LLM_MODEL']
        base_url = args.llm_base_url or os.environ.get('LLM_BASE_URL', None)
        api_version = os.environ.get('LLM_API_VERSION', None)
        llm_num_retries = int(os.environ.get('LLM_NUM_RETRIES', '4'))
        llm_retry_min_wait = int(os.environ.get('LLM_RETRY_MIN_WAIT', '5'))
        llm_retry_max_wait = int(os.environ.get('LLM_RETRY_MAX_WAIT', '30'))
        llm_retry_multiplier = int(os.environ.get('LLM_RETRY_MULTIPLIER', 2))
        llm_timeout = int(os.environ.get('LLM_TIMEOUT', 0))

        # Create LLMConfig instance
        llm_config = LLMConfig(
            model=model,
            api_key=SecretStr(api_key) if api_key else None,
            base_url=base_url,
            num_retries=llm_num_retries,
            retry_min_wait=llm_retry_min_wait,
            retry_max_wait=llm_retry_max_wait,
            retry_multiplier=llm_retry_multiplier,
            timeout=llm_timeout,
        )

        if api_version is not None:
            llm_config.api_version = api_version

        repo_instruction = None
        if args.repo_instruction_file:
            with open(args.repo_instruction_file, 'r') as f:
                repo_instruction = f.read()

        issue_type = args.issue_type

        # Read the prompt template
        prompt_file = args.prompt_file
        if prompt_file is None:
            if issue_type == 'issue':
                prompt_file = os.path.join(
                    os.path.dirname(__file__), 'prompts/resolve/basic-with-tests.jinja'
                )
            else:
                prompt_file = os.path.join(
                    os.path.dirname(__file__), 'prompts/resolve/basic-followup.jinja'
                )
        with open(prompt_file, 'r') as f:
            prompt_template = f.read()

        base_domain = args.base_domain
        if base_domain is None:
            base_domain = (
                'github.com' if platform == ProviderType.GITHUB else 'gitlab.com'
            )

        self.owner = owner
        self.repo = repo
        self.platform = platform
        self.max_iterations = args.max_iterations
        self.output_dir = args.output_dir
        self.llm_config = llm_config
        self.prompt_template = prompt_template
        self.issue_type = issue_type
        self.repo_instruction = repo_instruction
        self.issue_number = args.issue_number
        self.comment_id = args.comment_id

        factory = IssueHandlerFactory(
            owner=self.owner,
            repo=self.repo,
            token=token,
            username=username,
            platform=self.platform,
            base_domain=base_domain,
            issue_type=self.issue_type,
            llm_config=self.llm_config,
        )
        self.issue_handler = factory.create()

    @classmethod
    def _setup_sandbox_config(
        cls,
        base_container_image: str | None,
        runtime_container_image: str | None,
        is_experimental: bool,
    ) -> SandboxConfig:
        if runtime_container_image is not None and base_container_image is not None:
            raise ValueError('Cannot provide both runtime and base container images.')

        if (
            runtime_container_image is None
            and base_container_image is None
            and not is_experimental
        ):
            runtime_container_image = (
                f'ghcr.io/all-hands-ai/runtime:{openhands.__version__}-nikolaik'
            )

        # Convert container image values to string or None
        container_base = (
            str(base_container_image) if base_container_image is not None else None
        )
        container_runtime = (
            str(runtime_container_image)
            if runtime_container_image is not None
            else None
        )

        sandbox_config = SandboxConfig(
            base_container_image=container_base,
            runtime_container_image=container_runtime,
            enable_auto_lint=False,
            use_host_network=False,
            timeout=300,
        )

        # Configure sandbox for GitLab CI environment
        if cls.GITLAB_CI:
            sandbox_config.local_runtime_url = os.getenv(
                'LOCAL_RUNTIME_URL', 'http://localhost'
            )
            user_id = os.getuid() if hasattr(os, 'getuid') else 1000
            if user_id == 0:
                sandbox_config.user_id = get_unique_uid()

        return sandbox_config

    def initialize_runtime(self, runtime: Runtime) -> None:
        """Initialize the runtime for the agent."""
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Initialization')
        logger.info('-' * 30)
        obs: Observation

        action = CmdRunAction(command='cd /workspace')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to change directory to /workspace.\n{obs}')

        if self.platform == ProviderType.GITLAB and self.GITLAB_CI:
            action = CmdRunAction(command='sudo chown -R 1001:0 /workspace/*')
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        action = CmdRunAction(command='git config --global core.pager ""')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config.\n{obs}')

    async def complete_runtime(
        self,
        runtime: Runtime,
        base_commit: str,
    ) -> dict[str, Any]:
        """Complete the runtime for the agent."""
        logger.info('-' * 30)
        logger.info('BEGIN Runtime Completion')
        logger.info('-' * 30)
        obs: Observation

        action = CmdRunAction(command='cd /workspace')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(
                f'Failed to change directory to /workspace. Observation: {obs}'
            )

        action = CmdRunAction(command='git config --global core.pager ""')
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config. Observation: {obs}')

        action = CmdRunAction(
            command='git config --global --add safe.directory /workspace'
        )
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to set git config. Observation: {obs}')

        if self.platform == ProviderType.GITLAB and self.GITLAB_CI:
            action = CmdRunAction(command='sudo git add -A')
        else:
            action = CmdRunAction(command='git add -A')

        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if not isinstance(obs, CmdOutputObservation) or obs.exit_code != 0:
            raise RuntimeError(f'Failed to git add. Observation: {obs}')

        n_retries = 0
        git_patch = None
        while n_retries < 5:
            action = CmdRunAction(command=f'git diff --no-color --cached {base_commit}')
            action.set_hard_timeout(600 + 100 * n_retries)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})
            n_retries += 1
            if isinstance(obs, CmdOutputObservation):
                if obs.exit_code == 0:
                    git_patch = obs.content.strip()
                    break
                else:
                    logger.info('Failed to get git diff, retrying...')
                    await asyncio.sleep(10)
            elif isinstance(obs, ErrorObservation):
                logger.error(f'Error occurred: {obs.content}. Retrying...')
                await asyncio.sleep(10)
            else:
                raise ValueError(f'Unexpected observation type: {type(obs)}')

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
