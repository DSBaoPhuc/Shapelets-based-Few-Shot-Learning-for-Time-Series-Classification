class Shapelet:
    def __init__(self, data, start_pos, length, source_id, class_label):
        self.data = data
        self.start_pos = start_pos
        self.length = length
        self.source_id = source_id
        self.class_label = class_label
        self.quality = 0
        self.split_threshold = 0
        self.covered_instances = []
    
    def __lt__(self, other):
        return self.quality > other.quality 
    
    def set_quality(self, quality):
        self.quality = quality

    def set_split_threshold(self, threshold):
        self.split_threshold = threshold

    def add_covered_instance(self, instance_id):
        if instance_id not in self.covered_instances:
            self.covered_instances.append(instance_id)