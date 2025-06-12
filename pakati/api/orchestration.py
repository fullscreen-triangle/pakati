"""
API for the Pakati orchestration system.

This module provides API endpoints for accessing the orchestration system,
allowing users to create plans, execute tasks, and manage the generation process.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model_hub import ModelHub
from ..orchestration import Context, Planner, ReasoningEngine, Solver, IntuitiveChecker


# Model definitions
class GoalRequest(BaseModel):
    """Request to create a plan from a goal."""
    
    goal: str
    parameters: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    """Request to provide feedback on a plan."""
    
    feedback: str


class ModelRequest(BaseModel):
    """Request to get information about available models."""
    
    capability: Optional[str] = None
    provider: Optional[str] = None


# Create router
router = APIRouter(
    prefix="/orchestration",
    tags=["orchestration"],
)


# State
_context = None
_planner = None
_reasoning_engine = None
_solver = None
_intuitive_checker = None
_model_hub = None


def initialize_orchestration(api_keys: Optional[Dict[str, str]] = None) -> None:
    """
    Initialize the orchestration system.
    
    Args:
        api_keys: API keys for different model providers
    """
    global _context, _planner, _reasoning_engine, _solver, _intuitive_checker, _model_hub
    
    # Create model hub
    _model_hub = ModelHub(api_keys=api_keys)
    
    # Create context
    _context = Context(primary_goal="", persist=True)
    
    # Create orchestration components
    _planner = Planner(context=_context, model_hub=_model_hub)
    _reasoning_engine = ReasoningEngine(context=_context)
    _solver = Solver(context=_context)
    _intuitive_checker = IntuitiveChecker(context=_context)


# Endpoints
@router.post("/plans", response_model=Dict[str, Any])
def create_plan(request: GoalRequest) -> Dict[str, Any]:
    """
    Create a plan from a goal.
    
    Args:
        request: Plan creation request
        
    Returns:
        Created plan
    """
    # Ensure orchestration is initialized
    if _context is None:
        initialize_orchestration()
        
    # Update context with primary goal if it's empty
    if not _context.primary_goal:
        _context.primary_goal = request.goal
        
    # Set parameters and constraints if provided
    if request.parameters:
        for key, value in request.parameters.items():
            _context.set_parameter(key, value)
            
    if request.constraints:
        for key, value in request.constraints.items():
            _context.set_constraint(key, value)
            
    # Create plan
    plan = _planner.create_plan(request.goal)
    
    # Use reasoning engine to resolve any conflicts
    plan = _reasoning_engine.resolve_conflicts(plan)
    
    return plan.to_dict()


@router.get("/plans/{plan_id}", response_model=Dict[str, Any])
def get_plan(plan_id: str) -> Dict[str, Any]:
    """
    Get a plan by ID.
    
    Args:
        plan_id: ID of the plan
        
    Returns:
        Plan details
    """
    # Ensure orchestration is initialized
    if _planner is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    plan = _planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan with ID {plan_id} not found")
        
    return plan.to_dict()


@router.post("/plans/{plan_id}/feedback", response_model=Dict[str, Any])
def update_plan(plan_id: str, request: FeedbackRequest) -> Dict[str, Any]:
    """
    Update a plan based on feedback.
    
    Args:
        plan_id: ID of the plan
        request: Feedback request
        
    Returns:
        Updated plan
    """
    # Ensure orchestration is initialized
    if _planner is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    plan = _planner.update_plan(plan_id, request.feedback)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan with ID {plan_id} not found")
        
    # Use reasoning engine to resolve any conflicts
    plan = _reasoning_engine.resolve_conflicts(plan)
    
    return plan.to_dict()


@router.get("/plans/{plan_id}/tasks", response_model=List[Dict[str, Any]])
def get_next_tasks(plan_id: str) -> List[Dict[str, Any]]:
    """
    Get the next tasks to execute in a plan.
    
    Args:
        plan_id: ID of the plan
        
    Returns:
        List of tasks ready to be executed
    """
    # Ensure orchestration is initialized
    if _planner is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    tasks = _planner.get_next_tasks(plan_id)
    return [task.to_dict() for task in tasks]


@router.post("/plans/{plan_id}/tasks/{task_id}/execute", response_model=Dict[str, Any])
def execute_task(plan_id: str, task_id: str) -> Dict[str, Any]:
    """
    Execute a task.
    
    Args:
        plan_id: ID of the plan
        task_id: ID of the task
        
    Returns:
        Execution results
    """
    # Ensure orchestration is initialized
    if _planner is None or _reasoning_engine is None or _model_hub is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    # Get the plan and task
    plan = _planner.get_plan(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan with ID {plan_id} not found")
        
    task = plan.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task with ID {task_id} not found")
        
    # Optimize the task
    optimized_task = _reasoning_engine.optimize_task(task)
    
    # Get the executable model
    executable = _reasoning_engine.get_executable_model(optimized_task)
    
    # Execute the model
    try:
        result = _model_hub.execute_with_model(
            model_id=executable["model_name"],
            prompt=executable["prompt"],
            **executable["parameters"]
        )
        
        # Update task status
        _planner.update_task_status(plan_id, task_id, "completed", result)
        
        return {
            "task_id": task_id,
            "status": "completed",
            "result": result
        }
    except Exception as e:
        # Update task status
        _planner.update_task_status(plan_id, task_id, "failed")
        
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }


@router.get("/models", response_model=List[Dict[str, Any]])
def get_available_models(request: ModelRequest) -> List[Dict[str, Any]]:
    """
    Get information about available models.
    
    Args:
        request: Model request
        
    Returns:
        List of model information
    """
    # Ensure orchestration is initialized
    if _model_hub is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    return _model_hub.get_available_models(
        capability=request.capability,
        provider=request.provider
    )


@router.post("/solve", response_model=Dict[str, Any])
def solve_problem(problem_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve an optimization problem.
    
    Args:
        problem_type: Type of problem to solve
        data: Problem data
        
    Returns:
        Solution
    """
    # Ensure orchestration is initialized
    if _solver is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    return _solver.solve(problem_type, **data)


@router.post("/check-alignment", response_model=Dict[str, Any])
def check_alignment(image_data: str, goal: str) -> Dict[str, Any]:
    """
    Check if an image aligns with a goal.
    
    Args:
        image_data: Base64-encoded image data
        goal: Goal to check against
        
    Returns:
        Alignment information
    """
    # Ensure orchestration is initialized
    if _intuitive_checker is None:
        raise HTTPException(status_code=500, detail="Orchestration not initialized")
        
    import base64
    import io
    from PIL import Image
    import numpy as np
    
    # Decode image
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(image)
    
    # Check alignment
    return _intuitive_checker.check_alignment(image_array, goal) 