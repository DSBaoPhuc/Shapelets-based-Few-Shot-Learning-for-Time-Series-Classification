# class PruneAndCoverage:
#     """
#     Algorithm 8: Combined pruning and coverage algorithm
#     """
#     @staticmethod
#     def prune_and_cover(candidate_shapelets, coverage_param):
#         shapelets = []
#         instance_table = {}
        
#         size = len(candidate_shapelets)
#         i = 0
        
#         while i < size:
#             if len(instance_table) == 0:  # If instance table is empty
#                 break
            
#             select_shapelet = candidate_shapelets[i]
#             shapelets = PruneAndCoverage._update_coverage(select_shapelet, coverage_param, 
#                                                          instance_table, shapelets)
            
#             for j in range(i + 1, size):
#                 to_prune = candidate_shapelets[j]
#                 PruneAndCoverage._prune_shapelets(select_shapelet, to_prune)
            
#             size = len(candidate_shapelets)  # Update size after pruning
#             i += 1
        
#         return shapelets
    
#     @staticmethod
#     def _update_coverage(shapelet, coverage_param, instance_table, shapelets):
#         """Update coverage based on selected shapelet"""
#         # Implementation would depend on specific coverage strategy
#         # This is a placeholder for the actual implementation
#         shapelets.append(shapelet)
#         return shapelets
    
#     @staticmethod
#     def _prune_shapelets(select_shapelet, to_prune):
#         """Prune similar shapelets"""
#         # Implementation would depend on specific pruning strategy
#         # This is a placeholder for the actual implementation
#         pass



import numpy as np
from typing import List
from Discriminative_Shapelet_Transform.shapelet_uni import *
from Discriminative_Shapelet_Transform.Univariate.utils_uni import *

class ShapeletsPruning:
    @staticmethod
    def prune(candidate_shapelets):
        """
        Prune similar shapelets from the candidate set.
        
        Args:
            candidate_shapelets: List of Shapelet objects
            
        Returns:
            List of non-similar Shapelet objects
        """
        if not candidate_shapelets:
            return []
        
        no_similar_shapelets = []
        size = len(candidate_shapelets)
        is_pruned = [False] * size  # Mark pruned shapelets
        
        for i in range(size):
            if is_pruned[i]:
                continue  # Skip pruned shapelets
            
            select_shapelet = candidate_shapelets[i]
            no_similar_shapelets.append(select_shapelet)
            distance = select_shapelet.split_threshold
            
            for j in range(i + 1, size):
                if is_pruned[j]:
                    continue  # Skip pruned shapelets
                
                to_prune = candidate_shapelets[j]
                dist = subdist(select_shapelet.data, to_prune.data)
                
                # If shapelets are similar and same class, prune the second one
                if dist <= distance and select_shapelet.class_label == to_prune.class_label:
                    is_pruned[j] = True  # Mark as pruned
        
        return no_similar_shapelets

class ShapeletsCoverage:
    @staticmethod
    def cover(non_similar_shapelets, coverage_param):
        """
        Select diverse shapelets that cover different instances.
        
        Args:
            non_similar_shapelets: List of Shapelet objects
            coverage_param: Number of times each instance should be covered
            
        Returns:
            List of selected diverse Shapelet objects
        """
        shapelets = []
        instance_table = {}  # Track covered instances
        
        sorted_shapelets = sorted(non_similar_shapelets, key=lambda s: s.quality, reverse=True)
        
        for shapelet in sorted_shapelets:
            # Skip shapelets with no covered instances
            if not shapelet.covered_instances:
                continue
                
            # Check if this shapelet contributes to new coverage
            contributes_to_coverage = False
            for instance_id in shapelet.covered_instances:
                if instance_table.get(instance_id, 0) < coverage_param:
                    contributes_to_coverage = True
                    break  # Stop checking further; this shapelet is useful
            
            if contributes_to_coverage:
                # Add the shapelet and update coverage counts
                shapelets.append(shapelet)
                
                for instance_id in shapelet.covered_instances:
                    instance_table[instance_id] = instance_table.get(instance_id, 0) + 1
        
        return shapelets

class PruneAndCoverage:
    @staticmethod
    def prune_and_coverage(candidate_shapelets, coverage_param):
        """
        Optimized implementation of Algorithm 8: PruneAndCoverage
        
        Args:
            candidate_shapelets: List of Shapelet objects
            coverage_param: Coverage parameter σ
            
        Returns:
            Filtered list of shapelets
        """
        shapelets = []
        instance_table = {}
        
        sorted_shapelets = sorted(candidate_shapelets, key=lambda s: s.quality)
        
        # Cache distances to avoid redundant calculations
        size = len(sorted_shapelets)
        distance_cache = {}
        for i in range(size):
            for j in range(i + 1, size):
                pair = (i, j)
                distance_cache[pair] = subdist(sorted_shapelets[i].data, sorted_shapelets[j].data)
        
        # Iterate through candidate shapelets
        is_pruned = [False] * size
        for i in range(size):
            if is_pruned[i]:
                continue
            
            select_shapelet = sorted_shapelets[i]
            
            # Skip shapelets with no covered instances
            if not select_shapelet.covered_instances:
                continue
            
            # Check if this shapelet contributes to new coverage
            contributes_to_coverage = False
            for instance_id in select_shapelet.covered_instances:
                if instance_table.get(instance_id, 0) < coverage_param:
                    contributes_to_coverage = True
                    break  # Stop checking further; this shapelet is useful
            
            if not contributes_to_coverage:
                continue  # Skip this shapelet as it doesn't improve coverage
            
            shapelets.append(select_shapelet)
            
            # Update coverage
            for instance_id in select_shapelet.covered_instances:
                instance_table[instance_id] = instance_table.get(instance_id, 0) + 1
            
            # Prune similar shapelets
            for j in range(i + 1, size):
                if is_pruned[j]:
                    continue
                
                to_prune = sorted_shapelets[j]
                pair = (i, j) if i < j else (j, i)
                dist = distance_cache[pair]
                
                if dist <= select_shapelet.split_threshold and select_shapelet.class_label == to_prune.class_label:
                    is_pruned[j] = True
        
        return shapelets
