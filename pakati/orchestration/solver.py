"""
Solver module for the Pakati orchestration layer.

This module provides optimization solvers for various computational tasks
that are better handled by traditional algorithms than by neural networks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from scipy import optimize

from .context import Context


class Solver:
    """
    Solver for optimization problems in image generation.
    
    The Solver is responsible for:
    - Solving optimization problems efficiently with traditional algorithms
    - Finding optimal parameters for specific constraints
    - Performing mathematical operations that neural networks struggle with
    - Providing deterministic solutions to well-defined problems
    """
    
    def __init__(self, context: Context):
        """
        Initialize the solver.
        
        Args:
            context: The context for solving
        """
        self.context = context
        self.solvers = {
            "linear": self.solve_linear,
            "nonlinear": self.solve_nonlinear,
            "layout": self.solve_layout_optimization,
            "color": self.solve_color_optimization,
            "mask": self.solve_mask_optimization,
        }
        
    def solve(self, problem_type: str, **kwargs) -> Dict[str, Any]:
        """
        Solve an optimization problem.
        
        Args:
            problem_type: Type of problem to solve
            **kwargs: Additional parameters for the solver
            
        Returns:
            Dictionary with solution
        """
        # Log the solver call
        self.context.add_entry(
            entry_type="solver_started",
            content={"problem_type": problem_type},
            metadata=kwargs
        )
        
        # Call the appropriate solver
        solver = self.solvers.get(problem_type)
        if not solver:
            raise ValueError(f"Unsupported problem type: {problem_type}")
            
        solution = solver(**kwargs)
        
        # Log the solution
        self.context.add_entry(
            entry_type="solver_completed",
            content={"problem_type": problem_type, "solution": solution},
            metadata=kwargs
        )
        
        return solution
    
    def solve_linear(self, coefficients: List[float], constraints: List[Dict[str, Any]], 
                    objective: str = "minimize") -> Dict[str, Any]:
        """
        Solve a linear optimization problem.
        
        Args:
            coefficients: Coefficients of the objective function
            constraints: List of constraints
            objective: Whether to minimize or maximize the objective
            
        Returns:
            Dictionary with solution
        """
        # Convert inputs to numpy arrays
        c = np.array(coefficients)
        
        # Process constraints
        A_ub = []
        b_ub = []
        A_eq = []
        b_eq = []
        bounds = []
        
        for constraint in constraints:
            if constraint["type"] == "inequality":
                A_ub.append(constraint["coefficients"])
                b_ub.append(constraint["constant"])
            elif constraint["type"] == "equality":
                A_eq.append(constraint["coefficients"])
                b_eq.append(constraint["constant"])
            elif constraint["type"] == "bound":
                bounds.append((constraint["lower"], constraint["upper"]))
        
        # Convert to numpy arrays
        A_ub = np.array(A_ub) if A_ub else None
        b_ub = np.array(b_ub) if b_ub else None
        A_eq = np.array(A_eq) if A_eq else None
        b_eq = np.array(b_eq) if b_eq else None
        
        # Solve the linear programming problem
        if objective == "minimize":
            result = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        else:  # maximize
            result = optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
            result.fun = -result.fun  # Correct the objective value
        
        return {
            "success": result.success,
            "solution": result.x.tolist() if result.success else None,
            "objective_value": float(result.fun) if result.success else None,
            "message": result.message,
        }
    
    def solve_nonlinear(self, objective_function: Callable, 
                       initial_guess: List[float],
                       bounds: Optional[List[Tuple[float, float]]] = None,
                       constraints: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Solve a nonlinear optimization problem.
        
        Args:
            objective_function: The objective function to minimize
            initial_guess: Initial guess for the solution
            bounds: Bounds for the variables
            constraints: List of constraints
            
        Returns:
            Dictionary with solution
        """
        # Process constraints
        constraints_list = []
        if constraints:
            for constraint in constraints:
                if constraint["type"] == "inequality":
                    constraints_list.append({
                        "type": "ineq",
                        "fun": constraint["function"]
                    })
                elif constraint["type"] == "equality":
                    constraints_list.append({
                        "type": "eq",
                        "fun": constraint["function"]
                    })
        
        # Solve the nonlinear programming problem
        result = optimize.minimize(
            objective_function,
            np.array(initial_guess),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints_list if constraints_list else None
        )
        
        return {
            "success": result.success,
            "solution": result.x.tolist() if result.success else None,
            "objective_value": float(result.fun) if result.success else None,
            "message": result.message,
        }
    
    def solve_layout_optimization(self, regions: List[Dict[str, Any]], 
                               canvas_size: Tuple[int, int],
                               spacing: float = 10.0) -> Dict[str, Any]:
        """
        Optimize the layout of regions on a canvas.
        
        Args:
            regions: List of regions to position
            canvas_size: Size of the canvas (width, height)
            spacing: Minimum spacing between regions
            
        Returns:
            Dictionary with optimized positions
        """
        # Simple grid layout algorithm
        width, height = canvas_size
        
        # Calculate grid dimensions based on number of regions
        n_regions = len(regions)
        grid_size = int(np.ceil(np.sqrt(n_regions)))
        
        cell_width = width / grid_size
        cell_height = height / grid_size
        
        positions = []
        for i, region in enumerate(regions):
            row = i // grid_size
            col = i % grid_size
            
            region_width = region.get("width", cell_width - spacing)
            region_height = region.get("height", cell_height - spacing)
            
            # Center the region in its cell
            x = col * cell_width + (cell_width - region_width) / 2
            y = row * cell_height + (cell_height - region_height) / 2
            
            positions.append({
                "id": region.get("id", i),
                "x": x,
                "y": y,
                "width": region_width,
                "height": region_height
            })
        
        return {
            "success": True,
            "positions": positions,
            "grid_size": grid_size,
        }
    
    def solve_color_optimization(self, target_colors: List[List[float]], 
                              constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize color palettes or transformations.
        
        Args:
            target_colors: List of target colors in RGB or LAB format
            constraints: Constraints on the color optimization
            
        Returns:
            Dictionary with optimized colors
        """
        # Convert to numpy arrays
        colors = np.array(target_colors)
        
        # Check color constraints
        min_contrast = None
        max_colors = None
        color_space = "RGB"
        
        for constraint in constraints:
            if constraint["type"] == "contrast":
                min_contrast = constraint["value"]
            elif constraint["type"] == "palette_size":
                max_colors = constraint["value"]
            elif constraint["type"] == "color_space":
                color_space = constraint["value"]
        
        # Simple color optimization - cluster colors if needed
        if max_colors and len(colors) > max_colors:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=max_colors, random_state=0)
            kmeans.fit(colors)
            optimized_colors = kmeans.cluster_centers_
        else:
            optimized_colors = colors
            
        # Ensure contrast constraints if needed
        if min_contrast:
            # This is a simplified approach - in practice would be more complex
            for i in range(len(optimized_colors)):
                for j in range(i+1, len(optimized_colors)):
                    # Calculate color distance
                    distance = np.sqrt(np.sum((optimized_colors[i] - optimized_colors[j])**2))
                    
                    # If contrast is too low, adjust one of the colors
                    if distance < min_contrast:
                        # Move the second color away from the first
                        direction = optimized_colors[j] - optimized_colors[i]
                        if np.sum(direction**2) > 0:  # Avoid division by zero
                            direction = direction / np.sqrt(np.sum(direction**2))
                            optimized_colors[j] = optimized_colors[i] + direction * min_contrast
                            
                        # Clip to valid color range
                        optimized_colors[j] = np.clip(optimized_colors[j], 0, 1)
        
        return {
            "success": True,
            "colors": optimized_colors.tolist(),
            "color_space": color_space,
        }
    
    def solve_mask_optimization(self, source_masks: List[np.ndarray], 
                             target_mask: np.ndarray,
                             weights: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Optimize the combination of masks to match a target mask.
        
        Args:
            source_masks: List of source masks
            target_mask: Target mask to match
            weights: Initial weights for each source mask
            
        Returns:
            Dictionary with optimized weights
        """
        # Convert inputs to numpy arrays
        source_masks = [np.array(mask, dtype=float) for mask in source_masks]
        target_mask = np.array(target_mask, dtype=float)
        
        # Flatten masks for optimization
        flattened_target = target_mask.flatten()
        flattened_sources = [mask.flatten() for mask in source_masks]
        
        # Define objective function
        def objective(weights):
            combined = sum(w * mask for w, mask in zip(weights, flattened_sources))
            return np.sum((combined - flattened_target) ** 2)
        
        # Initial weights
        if weights is None:
            weights = [1.0 / len(source_masks)] * len(source_masks)
        
        # Bounds: weights should be between 0 and 1
        bounds = [(0, 1) for _ in range(len(source_masks))]
        
        # Solve the optimization problem
        result = optimize.minimize(
            objective,
            np.array(weights),
            method="L-BFGS-B",
            bounds=bounds
        )
        
        # Calculate final mask
        optimized_weights = result.x
        combined_mask = sum(w * mask for w, mask in zip(optimized_weights, source_masks))
        
        return {
            "success": result.success,
            "weights": optimized_weights.tolist(),
            "combined_mask": combined_mask.tolist() if result.success else None,
            "error": float(result.fun) if result.success else None,
        } 