"""
Evidence Graph module for Pakati.

This module implements a knowledge/evidence graph that tracks tasks, objectives,
and evidence to provide measurable optimization targets for the metacognitive
orchestrator. This enables the system to know if it's moving in the right direction.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Set, Callable
from uuid import UUID, uuid4
from enum import Enum
import networkx as nx
import numpy as np

from .references import ReferenceImage
from .delta_analysis import Delta, DeltaType


class ObjectiveType(Enum):
    """Types of objectives that can be tracked."""
    VISUAL_SIMILARITY = "visual_similarity"      # How similar to reference
    COMPOSITION_QUALITY = "composition_quality"  # Layout and arrangement
    COLOR_HARMONY = "color_harmony"              # Color relationships
    DETAIL_RICHNESS = "detail_richness"          # Level of detail
    STYLE_CONSISTENCY = "style_consistency"      # Style adherence
    SEMANTIC_ACCURACY = "semantic_accuracy"      # Content correctness
    GLOBAL_COHERENCE = "global_coherence"        # Overall image coherence
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"  # Meeting constraints


class EvidenceType(Enum):
    """Types of evidence that can be collected."""
    MEASUREMENT = "measurement"      # Quantitative measurements
    COMPARISON = "comparison"        # Comparative assessments
    DETECTION = "detection"          # Feature/object detection
    CLASSIFICATION = "classification" # Category classification
    REGRESSION = "regression"        # Continuous value prediction
    CONSTRAINT = "constraint"        # Constraint satisfaction


@dataclass
class Objective:
    """Represents a measurable objective with success criteria."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    objective_type: ObjectiveType = ObjectiveType.VISUAL_SIMILARITY
    description: str = ""
    target_value: float = 1.0  # Target score (0.0 to 1.0)
    current_value: float = 0.0  # Current achievement score
    weight: float = 1.0  # Importance weight in global optimization
    measurement_function: Optional[Callable] = None  # How to measure this objective
    dependencies: List[UUID] = field(default_factory=list)  # Other objectives this depends on
    constraints: Dict[str, Any] = field(default_factory=dict)  # Constraints for this objective
    is_critical: bool = False  # Must be satisfied for success
    progress_history: List[float] = field(default_factory=list)  # Historical progress
    
    @property
    def satisfaction_score(self) -> float:
        """Calculate how well this objective is satisfied (0.0 to 1.0)."""
        if self.target_value == 0:
            return 1.0 if self.current_value == 0 else 0.0
        return min(self.current_value / self.target_value, 1.0)
    
    @property
    def is_satisfied(self) -> bool:
        """Check if objective is satisfied."""
        return self.satisfaction_score >= 0.8  # 80% threshold
    
    def update_progress(self, new_value: float) -> float:
        """Update progress and return improvement delta."""
        old_value = self.current_value
        self.current_value = new_value
        self.progress_history.append(new_value)
        return new_value - old_value


@dataclass 
class Evidence:
    """Represents a piece of evidence collected during generation."""
    
    id: UUID = field(default_factory=uuid4)
    evidence_type: EvidenceType = EvidenceType.MEASUREMENT
    objective_id: UUID = field(default_factory=uuid4)  # Which objective this supports
    timestamp: float = field(default_factory=time.time)
    value: Any = None  # The evidence value
    confidence: float = 1.0  # Confidence in this evidence (0.0 to 1.0)
    source: str = ""  # Where this evidence came from
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate evidence after creation."""
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError("Evidence confidence must be between 0.0 and 1.0")


@dataclass
class Task:
    """Represents a task in the evidence graph with objectives and constraints."""
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    description: str = ""
    region_id: Optional[UUID] = None  # Associated canvas region
    objectives: List[UUID] = field(default_factory=list)  # Objectives for this task
    dependencies: List[UUID] = field(default_factory=list)  # Tasks this depends on
    status: str = "pending"  # pending, active, completed, failed
    priority: float = 1.0  # Task priority (higher = more important)
    estimated_effort: float = 1.0  # Estimated computational effort
    actual_effort: float = 0.0  # Actual effort spent
    completion_time: Optional[float] = None
    
    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (dependencies satisfied)."""
        # This would check if dependent tasks are completed
        # Simplified for now
        return self.status == "pending"
    
    @property 
    def efficiency_score(self) -> float:
        """Calculate task efficiency (results per effort)."""
        if self.actual_effort == 0:
            return 0.0
        # This would calculate based on objective improvements per effort
        return 1.0  # Placeholder


class EvidenceGraph:
    """
    Knowledge/Evidence graph that tracks tasks, objectives, and evidence
    to provide measurable optimization targets for iterative refinement.
    """
    
    def __init__(self, goal: str = ""):
        """Initialize the evidence graph."""
        self.goal = goal
        self.graph = nx.DiGraph()  # Directed graph for dependencies
        
        # Core data structures
        self.objectives: Dict[UUID, Objective] = {}
        self.evidence: Dict[UUID, Evidence] = {}
        self.tasks: Dict[UUID, Task] = {}
        
        # Optimization state
        self.global_objective_function: Optional[Callable] = None
        self.optimization_history: List[Dict[str, float]] = []
        self.convergence_threshold = 0.01
        self.stagnation_counter = 0
        self.max_stagnation = 3
        
        # Initialize graph structure
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the graph structure."""
        # Add nodes and edges will be added as objectives and tasks are created
        pass
    
    def add_objective(
        self,
        name: str,
        objective_type: ObjectiveType,
        description: str = "",
        target_value: float = 1.0,
        weight: float = 1.0,
        is_critical: bool = False,
        measurement_function: Optional[Callable] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Objective:
        """Add a new objective to the graph."""
        
        objective = Objective(
            name=name,
            objective_type=objective_type,
            description=description,
            target_value=target_value,
            weight=weight,
            is_critical=is_critical,
            measurement_function=measurement_function,
            constraints=constraints or {}
        )
        
        self.objectives[objective.id] = objective
        
        # Add to graph
        self.graph.add_node(
            f"obj_{objective.id}",
            type="objective",
            data=objective
        )
        
        print(f"Added objective: {name} (target: {target_value:.2f}, weight: {weight:.2f})")
        return objective
    
    def add_task(
        self,
        name: str,
        description: str = "",
        region_id: Optional[UUID] = None,
        priority: float = 1.0,
        estimated_effort: float = 1.0
    ) -> Task:
        """Add a new task to the graph."""
        
        task = Task(
            name=name,
            description=description,
            region_id=region_id,
            priority=priority,
            estimated_effort=estimated_effort
        )
        
        self.tasks[task.id] = task
        
        # Add to graph
        self.graph.add_node(
            f"task_{task.id}",
            type="task", 
            data=task
        )
        
        print(f"Added task: {name} (priority: {priority:.2f})")
        return task
    
    def link_task_to_objective(self, task_id: UUID, objective_id: UUID) -> None:
        """Link a task to an objective it should optimize."""
        if task_id in self.tasks and objective_id in self.objectives:
            self.tasks[task_id].objectives.append(objective_id)
            
            # Add edge in graph
            self.graph.add_edge(
                f"task_{task_id}",
                f"obj_{objective_id}",
                type="optimizes"
            )
            
            print(f"Linked task {self.tasks[task_id].name} to objective {self.objectives[objective_id].name}")
    
    def add_dependency(self, dependent_id: UUID, dependency_id: UUID, dep_type: str = "task") -> None:
        """Add a dependency relationship between tasks or objectives."""
        if dep_type == "task":
            if dependent_id in self.tasks and dependency_id in self.tasks:
                self.tasks[dependent_id].dependencies.append(dependency_id)
                self.graph.add_edge(
                    f"task_{dependency_id}",
                    f"task_{dependent_id}",
                    type="dependency"
                )
        elif dep_type == "objective":
            if dependent_id in self.objectives and dependency_id in self.objectives:
                self.objectives[dependent_id].dependencies.append(dependency_id)
                self.graph.add_edge(
                    f"obj_{dependency_id}",
                    f"obj_{dependent_id}",
                    type="dependency"
                )
    
    def collect_evidence(
        self,
        objective_id: UUID,
        evidence_type: EvidenceType,
        value: Any,
        confidence: float = 1.0,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Evidence:
        """Collect evidence for an objective."""
        
        evidence = Evidence(
            evidence_type=evidence_type,
            objective_id=objective_id,
            value=value,
            confidence=confidence,
            source=source,
            metadata=metadata or {}
        )
        
        self.evidence[evidence.id] = evidence
        
        # Update objective based on evidence
        if objective_id in self.objectives:
            self._update_objective_from_evidence(objective_id, evidence)
        
        return evidence
    
    def _update_objective_from_evidence(self, objective_id: UUID, evidence: Evidence) -> None:
        """Update an objective's current value based on new evidence."""
        objective = self.objectives[objective_id]
        
        # Simple weighted update (could be more sophisticated)
        if evidence.evidence_type == EvidenceType.MEASUREMENT:
            if isinstance(evidence.value, (int, float)):
                # Weighted average with confidence
                current_weight = len(objective.progress_history) + 1
                new_weight = evidence.confidence
                total_weight = current_weight + new_weight
                
                new_value = (
                    (objective.current_value * current_weight) + 
                    (evidence.value * new_weight)
                ) / total_weight
                
                objective.update_progress(new_value)
    
    def calculate_global_score(self) -> float:
        """Calculate the global optimization score."""
        if not self.objectives:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        critical_satisfied = True
        
        for objective in self.objectives.values():
            weighted_score = objective.satisfaction_score * objective.weight
            total_weighted_score += weighted_score
            total_weight += objective.weight
            
            # Check critical objectives
            if objective.is_critical and not objective.is_satisfied:
                critical_satisfied = False
        
        # If critical objectives not satisfied, heavily penalize
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        if not critical_satisfied:
            base_score *= 0.3  # Heavy penalty for unsatisfied critical objectives
        
        return base_score
    
    def get_optimization_priorities(self) -> List[Tuple[UUID, float]]:
        """Get prioritized list of objectives to focus on next."""
        priorities = []
        
        for obj_id, objective in self.objectives.items():
            # Calculate priority based on:
            # 1. How far from target (higher gap = higher priority)
            # 2. Weight/importance
            # 3. Whether it's critical
            # 4. Recent progress trend
            
            gap = max(0, objective.target_value - objective.current_value)
            criticality_multiplier = 2.0 if objective.is_critical else 1.0
            
            # Recent progress trend (negative if getting worse)
            trend = 0.0
            if len(objective.progress_history) >= 2:
                recent = objective.progress_history[-3:]
                trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0.0
            
            # Higher priority for: large gaps, high weight, critical, negative trends
            priority_score = (
                gap * objective.weight * criticality_multiplier * 
                (1.0 + max(0, -trend))  # Boost priority if trending down
            )
            
            priorities.append((obj_id, priority_score))
        
        # Sort by priority (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities
    
    def get_next_tasks(self, max_tasks: int = 3) -> List[Task]:
        """Get the next tasks to execute based on priorities and dependencies."""
        # Get ready tasks (dependencies satisfied)
        ready_tasks = [task for task in self.tasks.values() if task.is_ready]
        
        # Get optimization priorities
        obj_priorities = dict(self.get_optimization_priorities())
        
        # Score tasks based on:
        # 1. Priority of objectives they optimize
        # 2. Task priority
        # 3. Efficiency history
        task_scores = []
        
        for task in ready_tasks:
            score = task.priority
            
            # Add score from objectives this task optimizes
            for obj_id in task.objectives:
                if obj_id in obj_priorities:
                    score += obj_priorities[obj_id]
            
            # Factor in efficiency
            score *= task.efficiency_score if task.efficiency_score > 0 else 0.5
            
            task_scores.append((task, score))
        
        # Sort by score and return top tasks
        task_scores.sort(key=lambda x: x[1], reverse=True)
        return [task for task, score in task_scores[:max_tasks]]
    
    def update_from_deltas(self, deltas: List[Delta]) -> None:
        """Update evidence graph based on delta analysis results."""
        for delta in deltas:
            # Convert delta to evidence for relevant objectives
            
            if delta.delta_type == DeltaType.COLOR_MISMATCH:
                # Find color-related objectives
                color_objectives = [
                    obj for obj in self.objectives.values()
                    if obj.objective_type in [ObjectiveType.COLOR_HARMONY, ObjectiveType.VISUAL_SIMILARITY]
                ]
                
                for obj in color_objectives:
                    # Convert severity to evidence (inverted - high severity = low achievement)
                    evidence_value = max(0, 1.0 - delta.severity)
                    self.collect_evidence(
                        obj.id,
                        EvidenceType.MEASUREMENT,
                        evidence_value,
                        confidence=delta.confidence,
                        source="delta_analysis",
                        metadata={"delta_type": delta.delta_type.value, "description": delta.description}
                    )
            
            elif delta.delta_type == DeltaType.COMPOSITION_ISSUE:
                comp_objectives = [
                    obj for obj in self.objectives.values()
                    if obj.objective_type == ObjectiveType.COMPOSITION_QUALITY
                ]
                
                for obj in comp_objectives:
                    evidence_value = max(0, 1.0 - delta.severity)
                    self.collect_evidence(
                        obj.id,
                        EvidenceType.MEASUREMENT,
                        evidence_value,
                        confidence=delta.confidence,
                        source="delta_analysis"
                    )
            
            # Add more delta type mappings as needed
    
    def check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.optimization_history) < 3:
            return False
        
        # Check if recent scores are stable
        recent_scores = [h["global_score"] for h in self.optimization_history[-3:]]
        score_variance = np.var(recent_scores)
        
        if score_variance < self.convergence_threshold:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        return self.stagnation_counter >= self.max_stagnation
    
    def record_optimization_step(self, step_info: Dict[str, Any]) -> None:
        """Record information about an optimization step."""
        step_info["timestamp"] = time.time()
        step_info["global_score"] = self.calculate_global_score()
        
        # Add individual objective scores
        step_info["objective_scores"] = {
            obj.name: obj.satisfaction_score 
            for obj in self.objectives.values()
        }
        
        self.optimization_history.append(step_info)
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate a progress report for the current state."""
        report = {
            "goal": self.goal,
            "global_score": self.calculate_global_score(),
            "optimization_steps": len(self.optimization_history),
            "objectives": {},
            "tasks": {},
            "convergence_status": {
                "converged": self.check_convergence(),
                "stagnation_counter": self.stagnation_counter
            }
        }
        
        # Objective details
        for obj in self.objectives.values():
            report["objectives"][obj.name] = {
                "current_value": obj.current_value,
                "target_value": obj.target_value,
                "satisfaction_score": obj.satisfaction_score,
                "is_satisfied": obj.is_satisfied,
                "is_critical": obj.is_critical,
                "weight": obj.weight
            }
        
        # Task details
        for task in self.tasks.values():
            report["tasks"][task.name] = {
                "status": task.status,
                "priority": task.priority,
                "efficiency_score": task.efficiency_score,
                "objectives_count": len(task.objectives)
            }
        
        return report
    
    def save_graph(self, filepath: str) -> None:
        """Save the evidence graph to disk."""
        graph_data = {
            "goal": self.goal,
            "objectives": {},
            "tasks": {},
            "evidence": {},
            "optimization_history": self.optimization_history
        }
        
        # Serialize objectives
        for obj_id, obj in self.objectives.items():
            graph_data["objectives"][str(obj_id)] = {
                "name": obj.name,
                "objective_type": obj.objective_type.value,
                "description": obj.description,
                "target_value": obj.target_value,
                "current_value": obj.current_value,
                "weight": obj.weight,
                "is_critical": obj.is_critical,
                "progress_history": obj.progress_history,
                "dependencies": [str(d) for d in obj.dependencies]
            }
        
        # Serialize tasks
        for task_id, task in self.tasks.items():
            graph_data["tasks"][str(task_id)] = {
                "name": task.name,
                "description": task.description,
                "region_id": str(task.region_id) if task.region_id else None,
                "status": task.status,
                "priority": task.priority,
                "objectives": [str(o) for o in task.objectives],
                "dependencies": [str(d) for d in task.dependencies]
            }
        
        with open(filepath, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        print(f"Evidence graph saved to: {filepath}")
    
    def export_visualization_data(self) -> Dict[str, Any]:
        """Export data for visualization (e.g., for web dashboard)."""
        nodes = []
        edges = []
        
        # Add objective nodes
        for obj in self.objectives.values():
            nodes.append({
                "id": f"obj_{obj.id}",
                "label": obj.name,
                "type": "objective",
                "satisfaction_score": obj.satisfaction_score,
                "is_critical": obj.is_critical,
                "color": "red" if obj.is_critical and not obj.is_satisfied else "green" if obj.is_satisfied else "yellow"
            })
        
        # Add task nodes  
        for task in self.tasks.values():
            nodes.append({
                "id": f"task_{task.id}",
                "label": task.name,
                "type": "task",
                "status": task.status,
                "priority": task.priority,
                "color": "blue" if task.status == "active" else "gray" if task.status == "completed" else "orange"
            })
        
        # Add edges from graph
        for edge in self.graph.edges(data=True):
            edges.append({
                "source": edge[0],
                "target": edge[1], 
                "type": edge[2].get("type", "unknown")
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "global_score": self.calculate_global_score(),
            "optimization_history": self.optimization_history[-20:]  # Last 20 steps
        } 