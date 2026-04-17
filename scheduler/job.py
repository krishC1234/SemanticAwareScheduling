import time

class Job:
	def __init__(self, k, features, path):
		self.k = k
		self.path = path
		self.model_name = path.stem
		self.features = features
		self.assigned_gpus = 0
		self.path = path
		self.start_time = time.time()
		self.submitted = False
    
	def __str__(self):
		return (f"Job({self.model_name}, k={self.k:.2f}, "
			f"family={self.features.get('family', '?')}, "
			f"batch_size={self.features.get('batch_size', 0)}, "
			f"params={self.features.get('param_count', 0):,}, "
			f"gpus={self.assigned_gpus})")
    
	def __repr__(self):
		return self.__str__()