# class ShapeletsCoverage:
#     """
#     Algorithm 6: Select diverse shapelets that cover different instances
#     """
#     @staticmethod
#     def cover(non_similar_shapelets, coverage_param):
#         shapelets = []
#         instance_table = {}  # Track covered instances
        
#         size = len(non_similar_shapelets)
        
#         for i in range(size):
#             selected = False
#             instance_ids = non_similar_shapelets[i].covered_instances
            
#             for instance_id in instance_ids:
#                 if instance_id in instance_table:
#                     selected = True
#                     count = instance_table[instance_id]
                    
#                     if count < coverage_param:
#                         if count + 1 > coverage_param:
#                             del instance_table[instance_id]
#                         else:
#                             instance_table[instance_id] = count + 1
            
#             if selected:
#                 shapelets.append(non_similar_shapelets[i])
        
#         return shapelets

class ShapeletsCoverage:
    """
    Algorithm 6: Select diverse shapelets that cover different instances
    """
    @staticmethod
    def cover(non_similar_shapelets, coverage_param):
        shapelets = []
        instance_table = {}  # Track covered instances
        
        # Sắp xếp shapelets theo quality giảm dần
        sorted_shapelets = sorted(non_similar_shapelets, key=lambda s: s.quality, reverse=True)
        
        for shapelet in sorted_shapelets:
            add_shapelet = False
            instance_ids = shapelet.covered_instances
            
            for instance_id in instance_ids:
                # Nếu instance chưa được cover đủ coverage_param lần
                if instance_id not in instance_table or instance_table[instance_id] < coverage_param:
                    add_shapelet = True
                    # Tăng số lần cover cho instance này
                    instance_table[instance_id] = instance_table.get(instance_id, 0) + 1
            
            if add_shapelet & shapelet.get(instance_id) < coverage_param:
                shapelets.append(shapelet)
        
        return shapelets