from typing import Iterator, Dict

class SampleMetadata:
    def __init__(self, difficulty: float = 0.0, tags: list = None):
        self.difficulty = difficulty
        self.tags = tags or []
    
def pick_samples(potential_samples: Iterator[SampleMetadata], 
                tags_to_include: Dict[str, float], 
                difficulty_threshold: float):
    selected = []
    
    for sample in potential_samples:
        # Check difficulty threshold
        if sample.difficulty < difficulty_threshold:
            continue
            
        # Check if sample has required tags and calculate score
        score = 0
        has_required_tags = False
        
        for tag in sample.tags:
            if tag in tags_to_include:
                score += tags_to_include[tag]
                has_required_tags = True
        
        if has_required_tags:
            selected.append((sample, score))
    
    # Sort by score (highest first) and return samples
    selected.sort(key=lambda x: x[1], reverse=True)
    return [sample for sample, _ in selected]
