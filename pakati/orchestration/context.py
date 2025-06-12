"""
Context module for the Pakati orchestration layer.

This module provides context management for orchestrating image generation,
maintaining state and history to ensure coherent goal-directed generation.
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..utils import get_config


@dataclass
class ContextEntry:
    """An entry in the context history."""
    
    timestamp: float = field(default_factory=time.time)
    entry_type: str = "note"
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the entry to a dictionary."""
        return asdict(self)


class Context:
    """
    Context manager for orchestrating the image generation process.
    
    The Context class maintains state and history throughout the generation
    process, including:
    
    - The primary goal/intention of the user
    - The current state of the generation
    - The history of operations and reasoning
    - Parameters and constraints for generation
    """
    
    def __init__(
        self,
        primary_goal: str = "",
        working_dir: Optional[str] = None,
        persist: bool = True,
    ):
        """
        Initialize the context.
        
        Args:
            primary_goal: The primary goal/intention of the user
            working_dir: Directory to store context files
            persist: Whether to persist context to disk
        """
        self.primary_goal = primary_goal
        self.history: List[ContextEntry] = []
        self.parameters: Dict[str, Any] = {}
        self.constraints: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {"stage": "initialization"}
        self.persist = persist
        
        # Set up working directory
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            storage_path = get_config("STORAGE_PATH", "./storage")
            self.working_dir = Path(storage_path) / f"context_{int(time.time())}"
            
        if self.persist:
            os.makedirs(self.working_dir, exist_ok=True)
            
        # Add initial entry
        self.add_entry(
            entry_type="goal",
            content=primary_goal,
            metadata={"timestamp": time.time()}
        )
            
    def add_entry(
        self,
        entry_type: str,
        content: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ContextEntry:
        """
        Add an entry to the context history.
        
        Args:
            entry_type: Type of entry (e.g., "goal", "decision", "operation")
            content: Content of the entry
            metadata: Additional metadata for the entry
            
        Returns:
            The created entry
        """
        entry = ContextEntry(
            timestamp=time.time(),
            entry_type=entry_type,
            content=content,
            metadata=metadata or {},
        )
        self.history.append(entry)
        
        # Persist to disk if enabled
        if self.persist:
            self._persist_entry(entry)
            
        return entry
    
    def update_state(self, **kwargs) -> None:
        """
        Update the current state.
        
        Args:
            **kwargs: Key-value pairs to update in the state
        """
        self.state.update(kwargs)
        
        # Add entry for significant state changes
        if "stage" in kwargs:
            self.add_entry(
                entry_type="state_change",
                content={"new_stage": kwargs["stage"]},
                metadata={"prev_stage": self.state.get("stage", None)}
            )
            
        # Persist to disk if enabled
        if self.persist:
            self._persist_state()
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set a parameter for generation.
        
        Args:
            key: Parameter key
            value: Parameter value
        """
        self.parameters[key] = value
        
        # Add entry for parameter changes
        self.add_entry(
            entry_type="parameter",
            content={key: value},
            metadata={"prev_value": self.parameters.get(key, None)}
        )
        
        # Persist to disk if enabled
        if self.persist:
            self._persist_parameters()
    
    def set_constraint(self, key: str, value: Any) -> None:
        """
        Set a constraint for generation.
        
        Args:
            key: Constraint key
            value: Constraint value
        """
        self.constraints[key] = value
        
        # Add entry for constraint changes
        self.add_entry(
            entry_type="constraint",
            content={key: value},
            metadata={"prev_value": self.constraints.get(key, None)}
        )
        
        # Persist to disk if enabled
        if self.persist:
            self._persist_constraints()
    
    def get_recent_entries(self, entry_type: Optional[str] = None, limit: int = 10) -> List[ContextEntry]:
        """
        Get recent entries from history.
        
        Args:
            entry_type: Filter by entry type
            limit: Maximum number of entries to return
            
        Returns:
            List of recent entries
        """
        if entry_type:
            filtered = [e for e in self.history if e.entry_type == entry_type]
            return filtered[-limit:]
        else:
            return self.history[-limit:]
        
    def get_primary_goal(self) -> str:
        """
        Get the primary goal/intention.
        
        Returns:
            The primary goal
        """
        return self.primary_goal
    
    def export(self, filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Export the context as a dictionary.
        
        Args:
            filepath: Path to save the export (optional)
            
        Returns:
            The context as a dictionary
        """
        context_dict = {
            "primary_goal": self.primary_goal,
            "state": self.state,
            "parameters": self.parameters,
            "constraints": self.constraints,
            "history": [entry.to_dict() for entry in self.history],
        }
        
        if filepath:
            with open(filepath, "w") as f:
                json.dump(context_dict, f, indent=2)
                
        return context_dict
    
    def _persist_entry(self, entry: ContextEntry) -> None:
        """Persist an entry to disk."""
        history_file = self.working_dir / "history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
            
    def _persist_state(self) -> None:
        """Persist the current state to disk."""
        state_file = self.working_dir / "state.json"
        with open(state_file, "w") as f:
            json.dump(self.state, f, indent=2)
            
    def _persist_parameters(self) -> None:
        """Persist parameters to disk."""
        params_file = self.working_dir / "parameters.json"
        with open(params_file, "w") as f:
            json.dump(self.parameters, f, indent=2)
            
    def _persist_constraints(self) -> None:
        """Persist constraints to disk."""
        constraints_file = self.working_dir / "constraints.json"
        with open(constraints_file, "w") as f:
            json.dump(self.constraints, f, indent=2)
            
    @classmethod
    def load(cls, directory: str) -> "Context":
        """
        Load a context from disk.
        
        Args:
            directory: Directory containing context files
            
        Returns:
            Loaded Context object
        """
        directory_path = Path(directory)
        
        # Load primary goal
        with open(directory_path / "history.jsonl", "r") as f:
            first_line = f.readline().strip()
            first_entry = json.loads(first_line)
            primary_goal = first_entry.get("content", "")
            
        # Create context
        context = cls(primary_goal=primary_goal, working_dir=directory, persist=True)
        
        # Load state
        state_file = directory_path / "state.json"
        if state_file.exists():
            with open(state_file, "r") as f:
                context.state = json.load(f)
                
        # Load parameters
        params_file = directory_path / "parameters.json"
        if params_file.exists():
            with open(params_file, "r") as f:
                context.parameters = json.load(f)
                
        # Load constraints
        constraints_file = directory_path / "constraints.json"
        if constraints_file.exists():
            with open(constraints_file, "r") as f:
                context.constraints = json.load(f)
                
        # Load history
        context.history = []
        history_file = directory_path / "history.jsonl"
        if history_file.exists():
            with open(history_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entry_dict = json.loads(line)
                        entry = ContextEntry(
                            timestamp=entry_dict.get("timestamp", time.time()),
                            entry_type=entry_dict.get("entry_type", "note"),
                            content=entry_dict.get("content"),
                            metadata=entry_dict.get("metadata", {})
                        )
                        context.history.append(entry)
                        
        return context 