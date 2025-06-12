"""
Planner module for the Pakati orchestration layer.

This module provides planning functionality for orchestrating image generation,
converting high-level user goals into concrete actionable plans and tasks.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .context import Context
from ..utils import get_config
from ..model_hub import ModelHub


@dataclass
class Task:
    """A specific image generation task to be performed."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    task_type: str = "generation"  # generation, modification, refinement, etc.
    region: Optional[List[Tuple[int, int]]] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Any = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "region": self.region,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model_name": self.model_name,
            "parameters": self.parameters,
            "dependencies": self.dependencies,
            "status": self.status,
        }


@dataclass
class Plan:
    """A plan consisting of multiple tasks to achieve a goal."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    goal: str = ""
    tasks: List[Task] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed
    feedback: str = ""
    
    def add_task(self, task: Task) -> None:
        """Add a task to the plan."""
        self.tasks.append(task)
        
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by its ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def get_next_tasks(self) -> List[Task]:
        """Get the next tasks that can be executed."""
        next_tasks = []
        
        for task in self.tasks:
            if task.status == "pending":
                # Check if all dependencies are completed
                deps_completed = True
                for dep_id in task.dependencies:
                    dep_task = self.get_task(dep_id)
                    if not dep_task or dep_task.status != "completed":
                        deps_completed = False
                        break
                        
                if deps_completed:
                    next_tasks.append(task)
                    
        return next_tasks
    
    def is_completed(self) -> bool:
        """Check if the plan is completed."""
        return all(task.status == "completed" for task in self.tasks)
    
    def update_task_status(self, task_id: str, status: str, result: Any = None) -> None:
        """Update the status of a task."""
        task = self.get_task(task_id)
        if task:
            task.status = status
            if result is not None:
                task.result = result
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "goal": self.goal,
            "tasks": [task.to_dict() for task in self.tasks],
            "dependencies": self.dependencies,
            "status": self.status,
            "feedback": self.feedback,
        }


class Planner:
    """
    Planner for orchestrating image generation tasks.
    
    The Planner is responsible for:
    - Converting high-level user goals into concrete plans
    - Breaking down plans into tasks with dependencies
    - Tracking the execution of plans and tasks
    - Adapting plans based on feedback
    """
    
    def __init__(self, context: Context, model_hub: Optional[ModelHub] = None, llm_interface=None):
        """
        Initialize the planner.
        
        Args:
            context: The context for planning
            model_hub: Hub for accessing AI models
            llm_interface: Interface to language model for plan generation
        """
        self.context = context
        self.model_hub = model_hub
        self.llm_interface = llm_interface
        self.plans: Dict[str, Plan] = {}
        
    def create_plan(self, goal: str) -> Plan:
        """
        Create a plan from a high-level goal.
        
        Args:
            goal: The high-level goal to achieve
            
        Returns:
            A plan consisting of tasks
        """
        # Log the planning process
        self.context.add_entry(
            entry_type="planning_started",
            content={"goal": goal},
            metadata={"timestamp": self.context.state.get("timestamp", None)}
        )
        
        # Create a basic plan structure
        plan = Plan(
            name=f"Plan for {goal[:30]}..." if len(goal) > 30 else f"Plan for {goal}",
            description=f"Plan to achieve: {goal}",
            goal=goal,
        )
        
        # Use LLM to generate tasks if available
        if self.llm_interface:
            tasks = self._generate_tasks_with_llm(goal)
            for task in tasks:
                plan.add_task(task)
        else:
            # Fall back to basic planning
            tasks = self._generate_basic_tasks(goal)
            for task in tasks:
                plan.add_task(task)
        
        # Store the plan
        self.plans[plan.id] = plan
        
        # Log the created plan
        self.context.add_entry(
            entry_type="plan_created",
            content=plan.to_dict(),
            metadata={"plan_id": plan.id}
        )
        
        return plan
    
    def _generate_tasks_with_llm(self, goal: str) -> List[Task]:
        """
        Generate tasks using a language model.
        
        Args:
            goal: The high-level goal
            
        Returns:
            List of tasks
        """
        # Request a plan from the LLM
        response = self.llm_interface.generate_plan(
            goal=goal,
            primary_goal=self.context.get_primary_goal(),
            context_history=self.context.get_recent_entries(limit=20),
            parameters=self.context.parameters,
            constraints=self.context.constraints,
        )
        
        # Process the response into tasks
        tasks = []
        for task_data in response.get("tasks", []):
            # Determine the best model for the task if not specified
            model_name = task_data.get("model_name")
            if not model_name and self.model_hub:
                task_type = task_data.get("task_type", "generation")
                required_capabilities = task_data.get("required_capabilities", [])
                preferred_provider = task_data.get("preferred_provider")
                constraints = task_data.get("constraints", {})
                
                model_name = self.model_hub.find_best_model(
                    task_type=task_type,
                    required_capabilities=required_capabilities,
                    preferred_provider=preferred_provider,
                    constraints=constraints
                )
            
            task = Task(
                name=task_data.get("name", ""),
                description=task_data.get("description", ""),
                task_type=task_data.get("task_type", "generation"),
                region=task_data.get("region"),
                prompt=task_data.get("prompt"),
                negative_prompt=task_data.get("negative_prompt"),
                model_name=model_name,
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", []),
            )
            tasks.append(task)
            
        return tasks
    
    def _generate_basic_tasks(self, goal: str) -> List[Task]:
        """
        Generate basic tasks without using a language model.
        
        Args:
            goal: The high-level goal
            
        Returns:
            List of tasks
        """
        # Parse the goal to identify regions and prompts
        # This is a simple fallback implementation
        
        # Default to a single full-canvas task if parsing fails
        image_size = int(get_config("IMAGE_SIZE", 1024))
        
        # Determine the best model for the task
        model_name = None
        if self.model_hub:
            model_name = self.model_hub.find_best_model(task_type="generation")
        else:
            model_name = get_config("DEFAULT_MODEL", "stable-diffusion-xl")
        
        task = Task(
            name="Generate full image",
            description=f"Generate an image based on: {goal}",
            task_type="generation",
            region=[(0, 0), (image_size, 0), (image_size, image_size), (0, image_size)],
            prompt=goal,
            model_name=model_name,
        )
        
        return [task]
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """
        Get a plan by ID.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            The plan, or None if not found
        """
        return self.plans.get(plan_id)
    
    def get_all_plans(self) -> List[Plan]:
        """
        Get all plans.
        
        Returns:
            List of all plans
        """
        return list(self.plans.values())
    
    def update_plan(self, plan_id: str, feedback: str) -> Optional[Plan]:
        """
        Update a plan based on feedback.
        
        Args:
            plan_id: ID of the plan
            feedback: Feedback on the plan
            
        Returns:
            The updated plan, or None if not found
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return None
        
        plan.feedback = feedback
        
        # Log the feedback
        self.context.add_entry(
            entry_type="plan_feedback",
            content={"feedback": feedback},
            metadata={"plan_id": plan_id}
        )
        
        # Use LLM to revise plan if available
        if self.llm_interface:
            # Get the revised plan
            response = self.llm_interface.revise_plan(
                plan=plan.to_dict(),
                feedback=feedback,
                context_history=self.context.get_recent_entries(limit=20),
            )
            
            # Update the plan with revised tasks
            if "tasks" in response:
                # Create a mapping of existing tasks
                existing_tasks = {task.id: task for task in plan.tasks}
                
                # Clear existing tasks
                plan.tasks = []
                
                # Add the revised tasks
                for task_data in response["tasks"]:
                    task_id = task_data.get("id")
                    
                    # Determine the best model for the task if not specified
                    model_name = task_data.get("model_name")
                    if not model_name and self.model_hub:
                        task_type = task_data.get("task_type", "generation")
                        required_capabilities = task_data.get("required_capabilities", [])
                        preferred_provider = task_data.get("preferred_provider")
                        constraints = task_data.get("constraints", {})
                        
                        model_name = self.model_hub.find_best_model(
                            task_type=task_type,
                            required_capabilities=required_capabilities,
                            preferred_provider=preferred_provider,
                            constraints=constraints
                        )
                    
                    if task_id and task_id in existing_tasks:
                        # Update existing task
                        task = existing_tasks[task_id]
                        task.name = task_data.get("name", task.name)
                        task.description = task_data.get("description", task.description)
                        task.task_type = task_data.get("task_type", task.task_type)
                        task.region = task_data.get("region", task.region)
                        task.prompt = task_data.get("prompt", task.prompt)
                        task.negative_prompt = task_data.get("negative_prompt", task.negative_prompt)
                        task.model_name = model_name or task.model_name
                        task.parameters = task_data.get("parameters", task.parameters)
                        task.dependencies = task_data.get("dependencies", task.dependencies)
                    else:
                        # Create new task
                        task = Task(
                            name=task_data.get("name", ""),
                            description=task_data.get("description", ""),
                            task_type=task_data.get("task_type", "generation"),
                            region=task_data.get("region"),
                            prompt=task_data.get("prompt"),
                            negative_prompt=task_data.get("negative_prompt"),
                            model_name=model_name,
                            parameters=task_data.get("parameters", {}),
                            dependencies=task_data.get("dependencies", []),
                        )
                    plan.tasks.append(task)
        
        # Log the plan update
        self.context.add_entry(
            entry_type="plan_updated",
            content=plan.to_dict(),
            metadata={"plan_id": plan_id}
        )
        
        return plan
    
    def get_next_tasks(self, plan_id: str) -> List[Task]:
        """
        Get the next tasks to execute in a plan.
        
        Args:
            plan_id: ID of the plan
            
        Returns:
            List of tasks ready to be executed
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return []
        
        return plan.get_next_tasks()
    
    def update_task_status(
        self, plan_id: str, task_id: str, status: str, result: Any = None
    ) -> None:
        """
        Update the status of a task.
        
        Args:
            plan_id: ID of the plan
            task_id: ID of the task
            status: New status of the task
            result: Result of the task (optional)
        """
        plan = self.get_plan(plan_id)
        if not plan:
            return
        
        plan.update_task_status(task_id, status, result)
        
        # Log the task update
        self.context.add_entry(
            entry_type="task_updated",
            content={"task_id": task_id, "status": status},
            metadata={"plan_id": plan_id}
        )
        
        # Check if all tasks are completed
        if plan.is_completed():
            plan.status = "completed"
            
            # Log the plan completion
            self.context.add_entry(
                entry_type="plan_completed",
                content={"plan_id": plan_id},
                metadata={"timestamp": self.context.state.get("timestamp", None)}
            ) 