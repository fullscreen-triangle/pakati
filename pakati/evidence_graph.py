"""
Evidence Graph module for Pakati.

This module implements a knowledge/evidence graph that tracks tasks, objectives,
and evidence to provide measurable optimization targets for the metacognitive
orchestrator. This solves the fundamental problem: how does the AI know it's
moving in the right direction toward the goal?
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
from PIL import Image

from .references import ReferenceImage
from .delta_analysis import Delta, DeltaType


class ObjectiveType(Enum):
    """Types of objectives that can be tracked and optimized."""
    VISUAL_SIMILARITY = "visual_similarity"      # Match reference images
    COMPOSITION_QUALITY = "composition_quality"  # Layout and spatial arrangement
    COLOR_HARMONY = "color_harmony"              # Color relationships and palette
    DETAIL_RICHNESS = "detail_richness"          # Level of detail and complexity
    STYLE_CONSISTENCY = "style_consistency"      # Artistic/photographic style
    SEMANTIC_ACCURACY = "semantic_accuracy"      # Content correctness
    GLOBAL_COHERENCE = "global_coherence"        # Overall image coherence
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"  # Meeting hard constraints


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
    """
    Represents a measurable objective in the optimization process.
    
    Unlike vague goals, objectives are specific, measurable, and have
    clear criteria for satisfaction using fuzzy logic.
    """
    
    id: UUID = field(default_factory=uuid4)
    name: str = ""
    objective_type: ObjectiveType = ObjectiveType.VISUAL_SIMILARITY
    description: str = ""
    target_value: float = 0.8  # What we're trying to achieve (0.0 to 1.0)
    tolerance: float = 0.1  # Acceptable range for fuzzy satisfaction
    weight: float = 1.0  # Importance weight in global objective function
    is_critical: bool = False  # Must be satisfied for success
    measurement_function: Optional[Callable] = None  # How to measure this
    
    # Fuzzy satisfaction tracking
    current_value: float = 0.0  # Current measured value
    satisfaction_degree: float = 0.0  # Fuzzy satisfaction (0.0 to 1.0)
    satisfaction_history: List[float] = field(default_factory=list)  # Track changes over time
    
    # Evidence collection
    evidence: List[Evidence] = field(default_factory=list)
    evidence_sources: Set[str] = field(default_factory=set)
    
    # Progress tracking
    baseline_value: float = 0.0  # Starting point
    best_value_achieved: float = 0.0  # Best value seen so far
    improvement_velocity: float = 0.0  # Rate of improvement
    stagnation_count: int = 0  # How many steps without improvement
    
    # Legacy fields for compatibility
    region_id: Optional[UUID] = None  # Which region this applies to
    reference_ids: List[UUID] = field(default_factory=list)  # Relevant references
    dependencies: Set[UUID] = field(default_factory=set)  # Dependencies on other objectives
    
    @property
    def is_satisfied(self) -> bool:
        """Legacy binary satisfaction check (fuzzy satisfaction > 0.7)."""
        return self.satisfaction_degree > 0.7
    
    @property
    def satisfaction_score(self) -> float:
        """Get the current fuzzy satisfaction score."""
        return self.satisfaction_degree
    
    def update_value(self, new_value: float) -> float:
        """Update current value and recalculate fuzzy satisfaction."""
        old_value = self.current_value
        self.current_value = new_value
        
        # Update fuzzy satisfaction
        self._update_fuzzy_satisfaction()
        
        return new_value - old_value
    
    def add_evidence(self, evidence: Evidence):
        """Add evidence and update satisfaction using fuzzy logic."""
        self.evidence.append(evidence)
        self.evidence_sources.add(evidence.source)
        
        # Update current value with confidence weighting
        if self.evidence:
            # Weighted average of recent evidence
            recent_evidence = self.evidence[-5:]  # Last 5 pieces of evidence
            weighted_sum = sum(e.value * e.confidence for e in recent_evidence)
            total_confidence = sum(e.confidence for e in recent_evidence)
            
            if total_confidence > 0:
                self.current_value = weighted_sum / total_confidence
        
        # Update satisfaction using fuzzy logic
        self._update_fuzzy_satisfaction()
        
        # Track improvement
        if self.current_value > self.best_value_achieved:
            self.best_value_achieved = self.current_value
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
    
    def _update_fuzzy_satisfaction(self):
        """Update fuzzy satisfaction degree based on current value and target."""
        # Import here to avoid circular imports
        from .iterative_refinement import FuzzyLogicEngine
        
        # Create a temporary fuzzy engine for satisfaction calculation
        fuzzy_engine = FuzzyLogicEngine()
        
        # Calculate fuzzy satisfaction
        self.satisfaction_degree = fuzzy_engine.evaluate_fuzzy_satisfaction(
            self.current_value, 
            self.target_value, 
            self.tolerance
        )
        
        # Add to history
        self.satisfaction_history.append(self.satisfaction_degree)
        
        # Calculate improvement velocity (recent trend)
        if len(self.satisfaction_history) >= 3:
            recent_scores = self.satisfaction_history[-3:]
            self.improvement_velocity = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
    def get_satisfaction_trend(self) -> str:
        """Get human-readable satisfaction trend."""
        if self.improvement_velocity > 0.05:
            return "improving"
        elif self.improvement_velocity < -0.05:
            return "declining" 
        else:
            return "stable"


@dataclass
class Evidence:
    """
    Represents a piece of evidence collected during generation.
    
    Evidence accumulates over time to inform the optimization process.
    """
    
    id: UUID = field(default_factory=uuid4)
    objective_id: UUID = field(default_factory=uuid4)
    timestamp: float = field(default_factory=time.time)
    value: Any = None  # The measured value
    confidence: float = 1.0  # How confident we are in this measurement
    source: str = ""  # Where this evidence came from
    metadata: Dict[str, Any] = field(default_factory=dict)


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


@dataclass 
class OptimizationStep:
    """Represents one step in the optimization process."""
    
    step_number: int = 0
    timestamp: float = field(default_factory=time.time)
    global_score: float = 0.0
    objective_scores: Dict[str, float] = field(default_factory=dict)
    actions_taken: List[str] = field(default_factory=list)
    evidence_collected: List[UUID] = field(default_factory=list)
    improvement_delta: float = 0.0


class EvidenceGraph:
    """
    Knowledge/Evidence graph that provides structured optimization targets.
    
    This is the missing piece that gives the metacognitive orchestrator
    tangible objectives to optimize, rather than blindly making adjustments.
    
    Think of it as a GPS for image generation - it knows:
    1. Where we are (current scores)
    2. Where we want to go (target objectives)  
    3. How to measure progress (evidence collection)
    4. What actions to prioritize (optimization function)
    """
    
    def __init__(self, goal: str = ""):
        """Initialize the evidence graph with a high-level goal."""
        self.goal = goal
        self.graph = nx.DiGraph()  # Directed graph for dependencies
        
        # Core data structures
        self.objectives: Dict[UUID, Objective] = {}
        self.evidence: Dict[UUID, Evidence] = {}
        self.tasks: Dict[UUID, Task] = {}
        
        # Optimization state
        self.global_objective_function: Optional[Callable] = None
        self.optimization_steps: List[OptimizationStep] = []
        self.convergence_threshold = 0.02  # When to consider converged
        self.stagnation_tolerance = 3  # Steps without improvement before adapting
        self.critical_objective_penalty = 0.5  # Penalty for unsatisfied critical objectives
        
        # Initialize graph structure
        self._initialize_graph()
    
    def _initialize_graph(self):
        """Initialize the graph structure."""
        # Add nodes and edges will be added as objectives and tasks are created
        pass
    
    def decompose_goal(self, goal: str, references: List[ReferenceImage] = None) -> List[Objective]:
        """
        Decompose a high-level goal into measurable objectives.
        
        This is crucial - it transforms vague goals like "make a beautiful landscape"
        into specific, measurable targets like "achieve 0.85 color harmony with 
        mountain reference".
        """
        objectives = []
        
        # Basic objectives that apply to most goals
        objectives.append(self.add_objective(
            "Global Coherence",
            ObjectiveType.GLOBAL_COHERENCE,
            target_value=0.8,
            weight=2.0,
            is_critical=True,
            description="Overall image should be coherent and well-composed"
        ))
        
        # Add reference-based objectives if references provided
        if references:
            for ref in references:
                for annotation in ref.annotations:
                    if annotation.aspect == "color":
                        objectives.append(self.add_objective(
                            f"Color Match: {annotation.description}",
                            ObjectiveType.COLOR_HARMONY,
                            target_value=0.85,
                            weight=1.5,
                            description=f"Match colors from: {annotation.description}",
                            reference_ids=[ref.id]
                        ))
                    
                    elif annotation.aspect == "composition":
                        objectives.append(self.add_objective(
                            f"Composition: {annotation.description}",
                            ObjectiveType.COMPOSITION_QUALITY,
                            target_value=0.8,
                            weight=1.8,
                            description=f"Match composition from: {annotation.description}",
                            reference_ids=[ref.id]
                        ))
                    
                    elif annotation.aspect == "style":
                        objectives.append(self.add_objective(
                            f"Style: {annotation.description}",
                            ObjectiveType.STYLE_CONSISTENCY,
                            target_value=0.75,
                            weight=1.2,
                            description=f"Match style from: {annotation.description}",
                            reference_ids=[ref.id]
                        ))
                    
                    elif annotation.aspect == "lighting":
                        objectives.append(self.add_objective(
                            f"Lighting: {annotation.description}",
                            ObjectiveType.VISUAL_SIMILARITY,
                            target_value=0.8,
                            weight=1.4,
                            description=f"Match lighting from: {annotation.description}",
                            reference_ids=[ref.id]
                        ))
        
        # Goal-specific objectives based on keywords
        goal_lower = goal.lower()
        
        if any(word in goal_lower for word in ["detailed", "intricate", "complex"]):
            objectives.append(self.add_objective(
                "Detail Richness",
                ObjectiveType.DETAIL_RICHNESS,
                target_value=0.85,
                weight=1.3,
                description="Generate rich detail and complexity"
            ))
        
        if any(word in goal_lower for word in ["portrait", "person", "face"]):
            objectives.append(self.add_objective(
                "Facial Accuracy", 
                ObjectiveType.SEMANTIC_ACCURACY,
                target_value=0.9,
                weight=2.0,
                is_critical=True,
                description="Accurate facial features and proportions"
            ))
        
        print(f"Decomposed goal '{goal}' into {len(objectives)} measurable objectives")
        return objectives
    
    def add_objective(
        self,
        name: str,
        objective_type: ObjectiveType,
        target_value: float = 1.0,
        weight: float = 1.0,
        is_critical: bool = False,
        description: str = "",
        region_id: Optional[UUID] = None,
        reference_ids: List[UUID] = None
    ) -> Objective:
        """Add a new measurable objective to the graph."""
        
        objective = Objective(
            name=name,
            objective_type=objective_type,
            target_value=target_value,
            weight=weight,
            is_critical=is_critical,
            description=description,
            region_id=region_id,
            reference_ids=reference_ids or []
        )
        
        self.objectives[objective.id] = objective
        print(f"  + Objective: {name} (target: {target_value:.2f}, weight: {weight:.1f})")
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
        value: Any,
        confidence: float = 1.0,
        source: str = "",
        metadata: Dict[str, Any] = None
    ) -> Evidence:
        """
        Collect evidence for an objective.
        
        This is how the system learns what's working and what isn't.
        """
        
        evidence = Evidence(
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
    
    def _update_objective_from_evidence(self, objective_id: UUID, evidence: Evidence):
        """Update an objective's current value based on new evidence."""
        objective = self.objectives[objective_id]
        
        if isinstance(evidence.value, (int, float)):
            # Weighted update considering confidence
            history_weight = len(objective.satisfaction_history) * 0.1 + 0.5  # Diminishing returns
            evidence_weight = evidence.confidence
            
            if objective.current_value == 0.0:  # First measurement
                new_value = evidence.value
            else:
                # Weighted average
                total_weight = history_weight + evidence_weight
                new_value = (
                    (objective.current_value * history_weight) + 
                    (evidence.value * evidence_weight)
                ) / total_weight
            
            improvement = objective.update_value(new_value)
            
            if improvement > 0.05:  # Significant improvement
                print(f"    ✓ {objective.name}: {objective.current_value:.3f} (+{improvement:.3f})")
            elif improvement < -0.05:  # Significant degradation
                print(f"    ✗ {objective.name}: {objective.current_value:.3f} ({improvement:.3f})")
    
    def update_from_deltas(self, deltas: List[Delta]) -> None:
        """
        Update evidence graph based on delta analysis results.
        
        This converts delta analysis into structured evidence.
        """
        
        for delta in deltas:
            # Convert severity to achievement score (inverted)
            evidence_value = max(0.0, 1.0 - delta.severity)
            
            # Map delta types to relevant objectives
            relevant_objectives = self._get_objectives_for_delta_type(delta.delta_type)
            
            for obj in relevant_objectives:
                self.collect_evidence(
                    obj.id,
                    evidence_value,
                    confidence=delta.confidence,
                    source="delta_analysis",
                    metadata={
                        "delta_type": delta.delta_type.value,
                        "description": delta.description,
                        "severity": delta.severity
                    }
                )
    
    def _get_objectives_for_delta_type(self, delta_type: DeltaType) -> List[Objective]:
        """Get objectives relevant to a specific delta type."""
        relevant = []
        
        type_mapping = {
            DeltaType.COLOR_MISMATCH: [ObjectiveType.COLOR_HARMONY, ObjectiveType.VISUAL_SIMILARITY],
            DeltaType.TEXTURE_DIFFERENCE: [ObjectiveType.DETAIL_RICHNESS, ObjectiveType.VISUAL_SIMILARITY],
            DeltaType.COMPOSITION_ISSUE: [ObjectiveType.COMPOSITION_QUALITY, ObjectiveType.GLOBAL_COHERENCE],
            DeltaType.LIGHTING_DIFFERENCE: [ObjectiveType.VISUAL_SIMILARITY, ObjectiveType.GLOBAL_COHERENCE],
            DeltaType.STYLE_MISMATCH: [ObjectiveType.STYLE_CONSISTENCY, ObjectiveType.VISUAL_SIMILARITY],
            DeltaType.DETAIL_MISSING: [ObjectiveType.DETAIL_RICHNESS, ObjectiveType.SEMANTIC_ACCURACY]
        }
        
        target_types = type_mapping.get(delta_type, [])
        
        for obj in self.objectives.values():
            if obj.objective_type in target_types:
                relevant.append(obj)
        
        return relevant
    
    def calculate_global_objective_function(self) -> float:
        """
        Calculate the global objective function value.
        
        This is THE key metric that tells us how well we're doing overall.
        It's what the optimization process tries to maximize.
        """
        if not self.objectives:
            return 0.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        critical_penalty = 1.0
        
        for objective in self.objectives.values():
            satisfaction = objective.satisfaction_score
            weighted_score = satisfaction * objective.weight
            
            total_weighted_score += weighted_score
            total_weight += objective.weight
            
            # Penalty for unsatisfied critical objectives
            if objective.is_critical and not objective.is_satisfied:
                critical_penalty *= self.critical_objective_penalty
        
        base_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        final_score = base_score * critical_penalty
        
        return final_score
    
    def get_optimization_priorities(self) -> List[Tuple[UUID, float, str]]:
        """
        Get prioritized list of objectives to focus on next.
        
        This tells the system what to work on for maximum impact.
        """
        priorities = []
        
        for obj_id, objective in self.objectives.items():
            # Calculate priority score based on:
            # 1. Gap from target (bigger gap = higher priority)
            # 2. Weight/importance
            # 3. Critical status
            # 4. Recent velocity (prioritize stalled objectives)
            
            gap = max(0, objective.target_value - objective.current_value)
            criticality_boost = 3.0 if objective.is_critical else 1.0
            
            # Boost priority for stalled objectives
            velocity_penalty = 1.0
            if objective.improvement_velocity < -0.01:  # Getting worse
                velocity_penalty = 2.0
            elif objective.improvement_velocity < 0.01:  # Stalled
                velocity_penalty = 1.5
            
            priority_score = gap * objective.weight * criticality_boost * velocity_penalty
            
            # Determine recommended action
            if gap > 0.3:
                action = "major_improvement"
            elif gap > 0.1:
                action = "minor_improvement" 
            elif objective.improvement_velocity < -0.01:
                action = "fix_degradation"
            else:
                action = "maintain"
            
            priorities.append((obj_id, priority_score, action))
        
        # Sort by priority score (highest first)
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities
    
    def should_continue_optimization(self) -> Tuple[bool, str]:
        """
        Determine if optimization should continue.
        
        Returns (should_continue, reason)
        """
        global_score = self.calculate_global_objective_function()
        
        # Check if all critical objectives are satisfied
        critical_satisfied = all(
            obj.is_satisfied for obj in self.objectives.values() 
            if obj.is_critical
        )
        
        if not critical_satisfied:
            return True, "critical_objectives_unsatisfied"
        
        # Check convergence
        if len(self.optimization_steps) >= 3:
            recent_scores = [step.global_score for step in self.optimization_steps[-3:]]
            score_variance = np.var(recent_scores)
            
            if score_variance < self.convergence_threshold:
                return False, "converged"
        
        # Check if we're making progress
        if len(self.optimization_steps) >= self.stagnation_tolerance:
            recent_improvements = [
                step.improvement_delta for step in self.optimization_steps[-self.stagnation_tolerance:]
            ]
            
            if all(imp <= 0.01 for imp in recent_improvements):
                return False, "stagnation"
        
        # Check if target score reached
        if global_score >= 0.9:
            return False, "target_achieved"
        
        return True, "optimization_continuing"
    
    def record_optimization_step(
        self,
        actions_taken: List[str] = None,
        evidence_collected: List[UUID] = None
    ) -> OptimizationStep:
        """Record an optimization step for tracking progress."""
        
        current_score = self.calculate_global_objective_function()
        previous_score = self.optimization_steps[-1].global_score if self.optimization_steps else 0.0
        improvement = current_score - previous_score
        
        step = OptimizationStep(
            step_number=len(self.optimization_steps) + 1,
            global_score=current_score,
            objective_scores={obj.name: obj.satisfaction_score for obj in self.objectives.values()},
            actions_taken=actions_taken or [],
            evidence_collected=evidence_collected or [],
            improvement_delta=improvement
        )
        
        self.optimization_steps.append(step)
        
        print(f"Optimization Step {step.step_number}: Global Score = {current_score:.3f} ({improvement:+.3f})")
        
        return step
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Generate a comprehensive progress report."""
        
        global_score = self.calculate_global_objective_function()
        continue_opt, reason = self.should_continue_optimization()
        
        # Categorize objectives by status
        satisfied = [obj for obj in self.objectives.values() if obj.is_satisfied]
        critical_unsatisfied = [obj for obj in self.objectives.values() if obj.is_critical and not obj.is_satisfied]
        needs_work = [obj for obj in self.objectives.values() if not obj.is_satisfied and not obj.is_critical]
        
        report = {
            "goal": self.goal,
            "global_score": global_score,
            "optimization_steps": len(self.optimization_steps),
            "should_continue": continue_opt,
            "stop_reason": reason,
            
            "objective_summary": {
                "total": len(self.objectives),
                "satisfied": len(satisfied),
                "critical_unsatisfied": len(critical_unsatisfied),
                "needs_work": len(needs_work)
            },
            
            "objectives": {
                obj.name: {
                    "current": obj.current_value,
                    "target": obj.target_value,
                    "satisfaction": obj.satisfaction_score,
                    "is_satisfied": obj.is_satisfied,
                    "is_critical": obj.is_critical,
                    "velocity": obj.improvement_velocity,
                    "weight": obj.weight
                }
                for obj in self.objectives.values()
            },
            
            "priorities": [
                {
                    "objective": self.objectives[obj_id].name,
                    "priority_score": score,
                    "action": action
                }
                for obj_id, score, action in self.get_optimization_priorities()[:5]
            ]
        }
        
        return report
    
    def get_actionable_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get specific, actionable recommendations for the next optimization step.
        
        This is what the metacognitive orchestrator uses to decide what to do next.
        """
        recommendations = []
        priorities = self.get_optimization_priorities()
        
        for obj_id, priority_score, action in priorities[:3]:  # Top 3 priorities
            objective = self.objectives[obj_id]
            
            recommendation = {
                "objective_name": objective.name,
                "objective_type": objective.objective_type.value,
                "priority_score": priority_score,
                "action_type": action,
                "current_score": objective.current_value,
                "target_score": objective.target_value,
                "gap": objective.target_value - objective.current_value,
                "region_id": objective.region_id,
                "reference_ids": objective.reference_ids,
                "is_critical": objective.is_critical
            }
            
            # Add specific suggestions based on objective type
            if objective.objective_type == ObjectiveType.COLOR_HARMONY:
                recommendation["suggestions"] = [
                    "Adjust color saturation and hue",
                    "Apply color correction based on reference",
                    "Enhance color relationships"
                ]
            elif objective.objective_type == ObjectiveType.COMPOSITION_QUALITY:
                recommendation["suggestions"] = [
                    "Reposition key elements",
                    "Adjust region boundaries",
                    "Improve spatial balance"
                ]
            elif objective.objective_type == ObjectiveType.DETAIL_RICHNESS:
                recommendation["suggestions"] = [
                    "Increase prompt detail and specificity",
                    "Add texture and complexity terms",
                    "Use higher guidance scale"
                ]
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def save_graph(self, filepath: str) -> None:
        """Save the evidence graph to disk."""
        graph_data = {
            "goal": self.goal,
            "objectives": {},
            "tasks": {},
            "evidence": {},
            "optimization_history": [step.__dict__ for step in self.optimization_steps]
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
                "satisfaction_history": obj.satisfaction_history,
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
            "global_score": self.calculate_global_objective_function(),
            "optimization_history": [step.__dict__ for step in self.optimization_steps[-20:]]  # Last 20 steps
        } 