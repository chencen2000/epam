from enum import Enum
from typing import List, Dict, Tuple


class TargetLabels(Enum):
    # Format: (value, index, weight)
    BACKGROUND = ("background", 0, 0.01)
    CONDENSATION = ("condensation", 1, 0.3)
    DIRT = ("dirt", 2, 0.6)
    SCRATCH = ("scratch", 3, 0.1)

    def __init__(self, value: str, index: int, weight: float):
        self.label = value
        self.index = index
        self.weight = weight

    @property
    def value(self) -> str:
        """Override value property to return the label string"""
        return self.label

    @classmethod
    def values(cls) -> List[str]:
        """Get all enum values as a list"""
        return [item.value for item in cls]
    
    @classmethod
    def names(cls) -> List[str]:
        """Get all enum names as a list"""
        return [item.name for item in cls]
    
    @classmethod
    def choices(cls) -> List[Tuple[str, str]]:
        """Get (value, name) pairs"""
        return [(item.value, item.name) for item in cls]
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if a value exists in the enum"""
        return value in cls.values()
    
    @classmethod
    def indices(cls) -> List[int]:
        """Get all indices as a list"""
        return [item.index for item in cls]
    
    @classmethod
    def weights(cls) -> List[float]:
        """Get all weights as a list"""
        return [item.weight for item in cls]
    
    @classmethod
    def get_by_index(cls, index: int) -> 'TargetLabels':
        """Get enum item by index"""
        for item in cls:
            if item.index == index:
                return item
        raise ValueError(f"No item found with index {index}")
    
    @classmethod
    def get_by_value(cls, value: str) -> 'TargetLabels':
        """Get enum item by value"""
        for item in cls:
            if item.value == value:
                return item
        raise ValueError(f"No item found with value '{value}'")
    
    @classmethod
    def index_to_weight_map(cls) -> Dict[int, float]:
        """Get mapping of index to weight"""
        return {item.index: item.weight for item in cls}
    
    @classmethod
    def value_to_index_map(cls) -> Dict[str, int]:
        """Get mapping of value to index"""
        return {item.value: item.index for item in cls}
    
    @classmethod
    def value_to_weight_map(cls) -> Dict[str, float]:
        """Get mapping of value to weight"""
        return {item.value: item.weight for item in cls}


# Example Usage
if __name__ == "__main__":
    print("Values:", TargetLabels.values())
    print("Names:", TargetLabels.names())
    print("Indices:", TargetLabels.indices())
    print("Weights:", TargetLabels.weights())
    print("Choices:", TargetLabels.choices())
    
    print("\nIndividual item properties:")
    for item in TargetLabels:
        print(f"{item.name}: value='{item.value}', index={item.index}, weight={item.weight}")
    
    print("\nLookup methods:")
    print("Get by index 2:", TargetLabels.get_by_index(2))
    print("Get by value 'dirt':", TargetLabels.get_by_value('dirt'))
    
    print("\nMapping methods:")
    print("Index to weight map:", TargetLabels.index_to_weight_map())
    print("Value to index map:", TargetLabels.value_to_index_map())
    print("Value to weight map:", TargetLabels.value_to_weight_map())
    
    print("\nValidation:")
    print("Has value 'pending':", TargetLabels.has_value('pending'))
    print("Has value 'scratch':", TargetLabels.has_value('scratch'))

    print(TargetLabels.BACKGROUND.value)