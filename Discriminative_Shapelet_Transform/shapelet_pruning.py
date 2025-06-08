from utils import subdist

class ShapeletsPruning:
    @staticmethod
    def prune(candidate_shapelets):
        if not candidate_shapelets:
            return []
        
        no_similar_shapelets = []
        size = len(candidate_shapelets)
        is_pruned = [False] * size  # mark shapelets as pruned
        
        for i in range(size):
            if is_pruned[i]:
                continue 
                
            select_shapelet = candidate_shapelets[i]
            no_similar_shapelets.append(select_shapelet)
            distance = select_shapelet.split_threshold
            
            for j in range(i + 1, size):
                if is_pruned[j]:
                    continue  # pass if already pruned
                    
                to_prune = candidate_shapelets[j]
                dist = subdist(select_shapelet.data, to_prune.data)
                
                # if similar or same class, remove the second shapelet
                if dist <= distance and select_shapelet.class_label == to_prune.class_label:
                    is_pruned[j] = True  # mark as pruned
        
        return no_similar_shapelets

# from utils import subdist

# class ShapeletsPruning:
#     @staticmethod
#     def prune(candidate_shapelets):
#         if not candidate_shapelets:
#             return []

#         size = len(candidate_shapelets)

#         while True:
#             no_similar_shapelets = []
#             for i in range(size):
#                 select_shapelet = candidate_shapelets[i]
#                 no_similar_shapelets.append(select_shapelet)
#                 distance = select_shapelet.split_threshold

#                 for j in range(i + 1, size):
#                     to_prune = candidate_shapelets[j]
#                     dist = subdist(select_shapelet.data, to_prune.data)

#                     if dist <= distance:
#                         if select_shapelet.class_label != to_prune.class_label:
#                             no_similar_shapelets.append(to_prune)
#                         # else: do not add (prune)
#                     else:
#                         no_similar_shapelets.append(to_prune)

#             # Update for next round
#             if len(no_similar_shapelets) == size:
#                 break  # no change → finish
#             candidate_shapelets = no_similar_shapelets
#             size = len(candidate_shapelets)

#         return no_similar_shapelets
