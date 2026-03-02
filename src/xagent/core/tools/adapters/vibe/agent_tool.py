"""
Agent Tool - Convert published agents into callable tools
"""

import logging
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type

from pydantic import BaseModel, Field

from .base import AbstractBaseTool, ToolCategory, ToolVisibility

logger = logging.getLogger(__name__)


class AgentToolArgs(BaseModel):
    """Arguments for agent tool."""

    task: str = Field(description="The task to delegate to the agent")


class AgentToolResult(BaseModel):
    """Result from agent tool execution."""

    response: str = Field(description="The agent's response")


class AgentTool(AbstractBaseTool):
    """
    Tool that wraps a published agent for execution.

    This allows published agents to be called as tools from other agents.
    """

    # Agent tools belong to the AGENT category
    category: ToolCategory = ToolCategory.AGENT

    def __init__(
        self,
        agent_id: int,
        agent_name: str,
        agent_description: str,
        db: Any,
        user_id: int,
        task_id: Optional[str] = None,
        workspace_base_dir: str = "uploads",
    ):
        """
        Initialize an agent tool.

        Args:
            agent_id: The database ID of the published agent
            agent_name: Name of the agent
            agent_description: Description of what this agent does
            db: Database session for loading agent config and models
            user_id: User ID for model access
            task_id: Task ID for workspace isolation
            workspace_base_dir: Base directory for workspace files
        """
        self._agent_id = agent_id
        self._agent_name = agent_name
        self._agent_description = agent_description
        self._db = db
        self._user_id = user_id
        self._task_id = task_id or f"agent_tool_{agent_id}"
        self._workspace_base_dir = workspace_base_dir
        self._visibility = ToolVisibility.PUBLIC

    @property
    def name(self) -> str:
        """Tool name."""
        return f"call_agent_{self._agent_name.lower().replace(' ', '_')}"

    @property
    def description(self) -> str:
        """Tool description."""
        return self._agent_description

    @property
    def tags(self) -> list[str]:
        """Tool tags."""
        return ["agent", "delegation"]

    def args_type(self) -> Type[BaseModel]:
        """Argument type."""
        return AgentToolArgs

    def return_type(self) -> Type[BaseModel]:
        """Return type."""
        return AgentToolResult

    def run_json_sync(self, args: Mapping[str, Any]) -> Any:
        """Sync execution not supported."""
        raise NotImplementedError("AgentTool only supports async execution.")

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        """Execute the agent with the given task."""
        import uuid

        from .....web.models.agent import Agent, AgentStatus
        from .....web.tools.config import WebToolConfig
        from .....web.user_isolated_memory import UserContext

        try:
            # Load agent from database
            agent = (
                self._db.query(Agent)
                .filter(
                    Agent.id == self._agent_id, Agent.status == AgentStatus.PUBLISHED
                )
                .first()
            )

            if not agent:
                return AgentToolResult(
                    response=f"Error: Agent {self._agent_id} not found or not published"
                ).model_dump()

            # Generate unique task ID for this execution
            execution_task_id = f"agent_{self._agent_id}_{uuid.uuid4().hex[:8]}"

            # Resolve models
            from .....core.agent.service import AgentService
            from .....core.memory.in_memory import InMemoryMemoryStore
            from .....web.services.llm_utils import UserAwareModelStorage

            storage = UserAwareModelStorage(self._db)
            default_llm = None
            fast_llm = None
            vision_llm = None
            compact_llm = None

            if agent.models:
                from .....web.models.model import Model as DBModel

                if agent.models.get("general"):
                    general_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["general"])
                        .first()
                    )
                    if general_model:
                        default_llm = storage.get_llm_by_name_with_access(
                            str(general_model.model_id), self._user_id
                        )

                if agent.models.get("small_fast"):
                    fast_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["small_fast"])
                        .first()
                    )
                    if fast_model:
                        fast_llm = storage.get_llm_by_name_with_access(
                            str(fast_model.model_id), self._user_id
                        )

                if agent.models.get("visual"):
                    visual_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["visual"])
                        .first()
                    )
                    if visual_model:
                        vision_llm = storage.get_llm_by_name_with_access(
                            str(visual_model.model_id), self._user_id
                        )

                if agent.models.get("compact"):
                    compact_model = (
                        self._db.query(DBModel)
                        .filter(DBModel.id == agent.models["compact"])
                        .first()
                    )
                    if compact_model:
                        compact_llm = storage.get_llm_by_name_with_access(
                            str(compact_model.model_id), self._user_id
                        )

            if not default_llm:
                return AgentToolResult(
                    response=f"Error: No valid model configured for agent {agent.name}"
                ).model_dump()

            # Create tool config with allowed collections, skills, and tools
            class MinimalRequest:
                def __init__(self, user_id: int):
                    self.user = type("obj", (), {"id": user_id})()

            tool_config = WebToolConfig(
                db=self._db,
                request=MinimalRequest(self._user_id),
                user_id=self._user_id,
                allowed_collections=agent.knowledge_bases
                if agent.knowledge_bases
                else None,
                allowed_skills=agent.skills if agent.skills else None,
                task_id=execution_task_id,
                workspace_base_dir=self._workspace_base_dir,
            )

            # Create agent service
            memory = InMemoryMemoryStore()
            agent_service = AgentService(
                name=agent.name,
                llm=default_llm,
                fast_llm=fast_llm,
                vision_llm=vision_llm,
                compact_llm=compact_llm,
                memory=memory,
                tool_config=tool_config,
                use_dag_pattern=True,
                id=execution_task_id,
                enable_workspace=True,
                workspace_base_dir=self._workspace_base_dir,
                task_id=execution_task_id,
                tracer=None,
            )

            # Build execution context
            execution_context: dict[str, Any] = {}
            if agent.instructions:
                execution_context["system_prompt"] = agent.instructions

            # Execute task
            with UserContext(self._user_id):
                result = await agent_service.execute_task(
                    task=args["task"],
                    context=execution_context if execution_context else None,
                    task_id=execution_task_id,
                )

            output = result.get("output", "No response generated")
            logger.info(
                f"Agent tool {self.name} executed successfully, output length: {len(output)}"
            )
            return AgentToolResult(response=output).model_dump()

        except Exception as e:
            error_msg = f"Error executing agent {self._agent_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return AgentToolResult(response=error_msg).model_dump()


def gen_agent_tool_name(agent_name: str) -> str:
    """
    Generate the tool name for a published agent.

    This is a centralized function to ensure consistent naming across the codebase.
    Tool name format: call_agent_{agent_name_lower_with_underscores}

    Args:
        agent_name: The name of the agent

    Returns:
        The tool name that will be used for this agent
    """
    return f"call_agent_{agent_name.lower().replace(' ', '_')}"


def get_published_agents_tools(
    db: Any,
    user_id: int,
    task_id: Optional[str] = None,
    workspace_base_dir: str = "uploads",
    excluded_agent_id: Optional[int] = None,
) -> list[AbstractBaseTool]:
    """
    Get tools for all published agents.

    Args:
        db: Database session
        user_id: User ID for model access
        task_id: Task ID for workspace isolation
        workspace_base_dir: Base directory for workspace files
        excluded_agent_id: Optional agent ID to exclude (to prevent self-calls)

    Returns:
        List of AgentTool instances
    """
    from .....web.models.agent import Agent, AgentStatus

    tools: list[AbstractBaseTool] = []

    try:
        # Query all published agents
        query = db.query(Agent).filter(
            Agent.status == AgentStatus.PUBLISHED,
            Agent.user_id == user_id,
        )

        # Exclude the specified agent (to prevent self-calls)
        if excluded_agent_id is not None:
            query = query.filter(Agent.id != excluded_agent_id)

        published_agents = query.all()

        logger.info(
            f"Found {len(published_agents)} published agents (excluded: {excluded_agent_id})"
        )

        for agent in published_agents:
            # Build description
            description = agent.description or f"Call {agent.name} agent"
            if agent.instructions:
                # Add brief instructions to description
                instructions_preview = agent.instructions[:200]
                if len(agent.instructions) > 200:
                    instructions_preview += "..."
                description += f". Instructions: {instructions_preview}"

            tool = AgentTool(
                agent_id=agent.id,
                agent_name=agent.name,
                agent_description=description,
                db=db,
                user_id=user_id,
                task_id=task_id,
                workspace_base_dir=workspace_base_dir,
            )
            tools.append(tool)
            logger.debug(f"Created agent tool: {tool.name}")

    except Exception as e:
        logger.error(f"Failed to load published agents as tools: {e}", exc_info=True)

    return tools


# Register tool creator for auto-discovery
# Import at bottom to avoid circular import with factory
from .factory import register_tool  # noqa: E402

if TYPE_CHECKING:
    from xagent.web.tools.config import WebToolConfig


@register_tool
async def create_agent_tools(config: "WebToolConfig") -> list[AbstractBaseTool]:
    """Create tools from published agents."""
    if not config.get_enable_agent_tools():
        return []

    try:
        db = config.get_db()
        user_id = config.get_user_id()
        if not user_id:
            return []

        excluded_agent_id = config.get_excluded_agent_id() if config else None

        return get_published_agents_tools(
            db=db,
            user_id=user_id,
            task_id=config.get_task_id(),
            workspace_base_dir="uploads",
            excluded_agent_id=excluded_agent_id,
        )
    except Exception as e:
        logger.warning(f"Failed to create agent tools: {e}")
        return []
