"""
Reasoning module for the Pakati orchestration layer.

This module provides reasoning functionality for orchestrating image generation,
converting plans into executable models with optimal parameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

from .context import Context
from .planner import Plan, Task


class ReasoningEngine:
    """
    Reasoning engine for orchestrating image generation.
    
    The ReasoningEngine is responsible for:
    - Converting plans into executable models with parameters
    - Optimizing parameters based on context and constraints
    - Making decisions about which models and techniques to use
    - Resolving conflicts and ambiguities in plans
    """
    
    def __init__(self, context: Context, llm_interface=None):
        """
        Initialize the reasoning engine.
        
        Args:
            context: The context for reasoning
            llm_interface: Interface to language model for reasoning
        """
        self.context = context
        self.llm_interface = llm_interface
        
    def analyze_plan(self, plan: Plan) -> Dict[str, Any]:
        """
        Analyze a plan and extract key information.
        
        Args:
            plan: The plan to analyze
            
        Returns:
            Dictionary of extracted information
        """
        # Log the analysis
        self.context.add_entry(
            entry_type="plan_analysis_started",
            content={"plan_id": plan.id},
            metadata={"goal": plan.goal}
        )
        
        # Extract information from the plan
        info = {
            "regions": self._extract_regions(plan),
            "prompts": self._extract_prompts(plan),
            "models": self._extract_models(plan),
            "dependencies": plan.dependencies,
            "has_conflicts": self._check_for_conflicts(plan),
        }
        
        # Log the analysis results
        self.context.add_entry(
            entry_type="plan_analysis_completed",
            content=info,
            metadata={"plan_id": plan.id}
        )
        
        return info
    
    def optimize_task(self, task: Task) -> Task:
        """
        Optimize a task for execution.
        
        Args:
            task: The task to optimize
            
        Returns:
            Optimized task
        """
        # Log the optimization
        self.context.add_entry(
            entry_type="task_optimization_started",
            content={"task_id": task.id},
            metadata={"task_type": task.task_type}
        )
        
        # Create a copy of the task to optimize
        optimized_task = Task(
            id=task.id,
            name=task.name,
            description=task.description,
            task_type=task.task_type,
            region=task.region,
            prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            model_name=task.model_name,
            parameters=task.parameters.copy(),
            dependencies=task.dependencies,
            status=task.status,
        )
        
        # Use LLM for optimization if available
        if self.llm_interface:
            optimized_params = self._optimize_parameters_with_llm(task)
            optimized_task.parameters.update(optimized_params)
        else:
            # Fall back to rule-based optimization
            self._optimize_parameters_rule_based(optimized_task)
        
        # Log the optimization results
        self.context.add_entry(
            entry_type="task_optimization_completed",
            content={"original": task.to_dict(), "optimized": optimized_task.to_dict()},
            metadata={"task_id": task.id}
        )
        
        return optimized_task
    
    def _extract_regions(self, plan: Plan) -> List[Dict[str, Any]]:
        """Extract regions from a plan."""
        regions = []
        for task in plan.tasks:
            if task.region:
                regions.append({
                    "task_id": task.id,
                    "region": task.region,
                    "task_type": task.task_type,
                })
        return regions
    
    def _extract_prompts(self, plan: Plan) -> Dict[str, str]:
        """Extract prompts from a plan."""
        prompts = {}
        for task in plan.tasks:
            if task.prompt:
                prompts[task.id] = task.prompt
        return prompts
    
    def _extract_models(self, plan: Plan) -> Dict[str, str]:
        """Extract models from a plan."""
        models = {}
        for task in plan.tasks:
            if task.model_name:
                models[task.id] = task.model_name
        return models
    
    def _check_for_conflicts(self, plan: Plan) -> bool:
        """Check for conflicts in a plan."""
        # Check for overlapping regions
        regions = []
        for task in plan.tasks:
            if task.region:
                regions.append((task.id, task.region))
        
        # Simple check for overlapping polygons
        # This is a simplified implementation
        for i, (id1, region1) in enumerate(regions):
            for j, (id2, region2) in enumerate(regions):
                if i != j and self._regions_overlap(region1, region2):
                    return True
        
        return False
    
    def _regions_overlap(self, region1, region2) -> bool:
        """Check if two regions overlap."""
        # This is a simplified check
        # A more accurate implementation would check for polygon intersection
        
        # Get bounding boxes
        def get_bbox(region):
            x_coords = [p[0] for p in region]
            y_coords = [p[1] for p in region]
            return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
        
        bbox1 = get_bbox(region1)
        bbox2 = get_bbox(region2)
        
        # Check for overlap
        if (bbox1[0] > bbox2[2] or bbox1[2] < bbox2[0] or
            bbox1[1] > bbox2[3] or bbox1[3] < bbox2[1]):
            return False
        return True
    
    def _optimize_parameters_with_llm(self, task: Task) -> Dict[str, Any]:
        """Optimize parameters using a language model."""
        # Request parameter optimization from the LLM
        response = self.llm_interface.optimize_parameters(
            task=task.to_dict(),
            context_history=self.context.get_recent_entries(limit=20),
            primary_goal=self.context.get_primary_goal(),
            parameters=self.context.parameters,
            constraints=self.context.constraints,
        )
        
        return response.get("parameters", {})
    
    def _optimize_parameters_rule_based(self, task: Task) -> None:
        """Optimize parameters using rule-based heuristics."""
        # Apply rule-based optimizations
        
        # Default parameters based on task type
        if task.task_type == "generation":
            if "steps" not in task.parameters:
                task.parameters["steps"] = 50
            if "guidance_scale" not in task.parameters:
                task.parameters["guidance_scale"] = 7.5
                
        elif task.task_type == "inpainting":
            if "steps" not in task.parameters:
                task.parameters["steps"] = 30
            if "guidance_scale" not in task.parameters:
                task.parameters["guidance_scale"] = 8.0
                
        elif task.task_type == "refinement":
            if "steps" not in task.parameters:
                task.parameters["steps"] = 20
            if "guidance_scale" not in task.parameters:
                task.parameters["guidance_scale"] = 9.0
                
        # Model-specific optimizations
        if task.model_name and "stable-diffusion" in task.model_name:
            if "sampler" not in task.parameters:
                task.parameters["sampler"] = "euler_a"
        elif task.model_name and "dall-e" in task.model_name:
            if "quality" not in task.parameters:
                task.parameters["quality"] = "standard"
                
        # Region-based optimizations
        if task.region:
            # Calculate region size
            def get_region_size(region):
                x_coords = [p[0] for p in region]
                y_coords = [p[1] for p in region]
                width = max(x_coords) - min(x_coords)
                height = max(y_coords) - min(y_coords)
                return width, height
                
            width, height = get_region_size(task.region)
            area = width * height
            
            # Adjust steps based on region size
            if area < 100000:  # Small region
                task.parameters["steps"] = min(task.parameters.get("steps", 50), 20)
            elif area > 500000:  # Large region
                task.parameters["steps"] = max(task.parameters.get("steps", 50), 60)
    
    def resolve_conflicts(self, plan: Plan) -> Plan:
        """
        Resolve conflicts in a plan.
        
        Args:
            plan: The plan with conflicts
            
        Returns:
            Plan with conflicts resolved
        """
        # Log the conflict resolution
        self.context.add_entry(
            entry_type="conflict_resolution_started",
            content={"plan_id": plan.id},
            metadata={"goal": plan.goal}
        )
        
        # Analyze the plan
        analysis = self.analyze_plan(plan)
        
        # If there are no conflicts, return the plan as is
        if not analysis.get("has_conflicts", False):
            return plan
        
        # Use LLM for conflict resolution if available
        if self.llm_interface:
            resolved_plan = self._resolve_conflicts_with_llm(plan, analysis)
        else:
            # Fall back to rule-based conflict resolution
            resolved_plan = self._resolve_conflicts_rule_based(plan, analysis)
        
        # Log the conflict resolution results
        self.context.add_entry(
            entry_type="conflict_resolution_completed",
            content={"original": plan.to_dict(), "resolved": resolved_plan.to_dict()},
            metadata={"plan_id": plan.id}
        )
        
        return resolved_plan
    
    def _resolve_conflicts_with_llm(self, plan: Plan, analysis: Dict[str, Any]) -> Plan:
        """Resolve conflicts using a language model."""
        # Request conflict resolution from the LLM
        response = self.llm_interface.resolve_conflicts(
            plan=plan.to_dict(),
            analysis=analysis,
            context_history=self.context.get_recent_entries(limit=20),
            primary_goal=self.context.get_primary_goal(),
        )
        
        # Create a new plan with resolved tasks
        resolved_plan = Plan(
            id=plan.id,
            name=plan.name,
            description=plan.description,
            goal=plan.goal,
            dependencies=plan.dependencies,
            status=plan.status,
            feedback=plan.feedback,
        )
        
        # Add the resolved tasks
        for task_data in response.get("tasks", []):
            task = Task(
                id=task_data.get("id", str(uuid.uuid4())),
                name=task_data.get("name", ""),
                description=task_data.get("description", ""),
                task_type=task_data.get("task_type", "generation"),
                region=task_data.get("region"),
                prompt=task_data.get("prompt"),
                negative_prompt=task_data.get("negative_prompt"),
                model_name=task_data.get("model_name"),
                parameters=task_data.get("parameters", {}),
                dependencies=task_data.get("dependencies", []),
                status=task_data.get("status", "pending"),
            )
            resolved_plan.add_task(task)
        
        return resolved_plan
    
    def _resolve_conflicts_rule_based(self, plan: Plan, analysis: Dict[str, Any]) -> Plan:
        """Resolve conflicts using rule-based heuristics."""
        # Create a new plan with resolved tasks
        resolved_plan = Plan(
            id=plan.id,
            name=plan.name,
            description=plan.description,
            goal=plan.goal,
            dependencies=plan.dependencies,
            status=plan.status,
            feedback=plan.feedback,
        )
        
        # Extract regions
        regions = analysis.get("regions", [])
        
        # Sort tasks by priority (generation first, then refinement, then modification)
        priority_order = {"generation": 0, "refinement": 1, "modification": 2, "inpainting": 3}
        
        sorted_tasks = sorted(
            plan.tasks,
            key=lambda t: priority_order.get(t.task_type, 10)
        )
        
        # Add tasks in priority order, skipping those that conflict with already added tasks
        added_regions = []
        for task in sorted_tasks:
            if not task.region:
                # If the task doesn't have a region, add it
                resolved_plan.add_task(task)
            elif not any(self._regions_overlap(task.region, added_region) for added_region in added_regions):
                # If the task's region doesn't overlap with any added regions, add it
                resolved_plan.add_task(task)
                added_regions.append(task.region)
            else:
                # Skip this task due to conflict
                pass
        
        return resolved_plan
    
    def get_executable_model(self, task: Task) -> Dict[str, Any]:
        """
        Convert a task into an executable model.
        
        Args:
            task: The task to convert
            
        Returns:
            Dictionary representing the executable model
        """
        # Log the model creation
        self.context.add_entry(
            entry_type="executable_model_creation",
            content={"task_id": task.id},
            metadata={"task_type": task.task_type}
        )
        
        # Optimize the task
        optimized_task = self.optimize_task(task)
        
        # Create executable model
        model = {
            "task_id": optimized_task.id,
            "task_type": optimized_task.task_type,
            "region": optimized_task.region,
            "prompt": optimized_task.prompt,
            "negative_prompt": optimized_task.negative_prompt,
            "model_name": optimized_task.model_name,
            "parameters": optimized_task.parameters,
        }
        
        # Add execution details
        if optimized_task.task_type == "generation":
            model["execution"] = {
                "function": "generate_image",
                "args": {
                    "prompt": optimized_task.prompt,
                    "negative_prompt": optimized_task.negative_prompt,
                    "model_name": optimized_task.model_name,
                    **optimized_task.parameters,
                }
            }
        elif optimized_task.task_type == "inpainting":
            model["execution"] = {
                "function": "inpaint_region",
                "args": {
                    "region": optimized_task.region,
                    "prompt": optimized_task.prompt,
                    "negative_prompt": optimized_task.negative_prompt,
                    "model_name": optimized_task.model_name,
                    **optimized_task.parameters,
                }
            }
        elif optimized_task.task_type == "refinement":
            model["execution"] = {
                "function": "refine_region",
                "args": {
                    "region": optimized_task.region,
                    "prompt": optimized_task.prompt,
                    "negative_prompt": optimized_task.negative_prompt,
                    "model_name": optimized_task.model_name,
                    **optimized_task.parameters,
                }
            }
        
        return model 