import random
import yaml
import os
from typing import Dict, List, Tuple
import numpy as np

class CombinedAI2THORDatasetSplitter:
    """Handles dataset splitting for combined iTHOR + RoboTHOR datasets"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # iTHOR scene definitions by room type
        self.ithor_scenes = {
            'kitchen': [f'FloorPlan{i}' for i in range(1, 31)],
            'living_room': [f'FloorPlan{i}' for i in range(201, 231)],
            'bedroom': [f'FloorPlan{i}' for i in range(301, 331)],
            'bathroom': [f'FloorPlan{i}' for i in range(401, 431)]
        }
        
        # RoboTHOR official splits
        self.robothor_splits = {
            'train': [
                'FloorPlan_Train1_1', 'FloorPlan_Train1_2', 'FloorPlan_Train1_3', 'FloorPlan_Train1_4', 'FloorPlan_Train1_5',
                'FloorPlan_Train2_1', 'FloorPlan_Train2_2', 'FloorPlan_Train2_3', 'FloorPlan_Train2_4', 'FloorPlan_Train2_5',
                'FloorPlan_Train3_1', 'FloorPlan_Train3_2', 'FloorPlan_Train3_3', 'FloorPlan_Train3_4', 'FloorPlan_Train3_5',
                'FloorPlan_Train4_1', 'FloorPlan_Train4_2', 'FloorPlan_Train4_3', 'FloorPlan_Train4_4', 'FloorPlan_Train4_5',
                'FloorPlan_Train5_1', 'FloorPlan_Train5_2', 'FloorPlan_Train5_3', 'FloorPlan_Train5_4', 'FloorPlan_Train5_5',
                'FloorPlan_Train6_1', 'FloorPlan_Train6_2', 'FloorPlan_Train6_3', 'FloorPlan_Train6_4', 'FloorPlan_Train6_5',
                'FloorPlan_Train7_1', 'FloorPlan_Train7_2', 'FloorPlan_Train7_3', 'FloorPlan_Train7_4', 'FloorPlan_Train7_5',
                'FloorPlan_Train8_1', 'FloorPlan_Train8_2', 'FloorPlan_Train8_3', 'FloorPlan_Train8_4', 'FloorPlan_Train8_5',
                'FloorPlan_Train9_1', 'FloorPlan_Train9_2', 'FloorPlan_Train9_3', 'FloorPlan_Train9_4', 'FloorPlan_Train9_5',
                'FloorPlan_Train10_1', 'FloorPlan_Train10_2', 'FloorPlan_Train10_3', 'FloorPlan_Train10_4', 'FloorPlan_Train10_5',
                'FloorPlan_Train11_1', 'FloorPlan_Train11_2', 'FloorPlan_Train11_3', 'FloorPlan_Train11_4', 'FloorPlan_Train11_5',
                'FloorPlan_Train12_1', 'FloorPlan_Train12_2', 'FloorPlan_Train12_3', 'FloorPlan_Train12_4', 'FloorPlan_Train12_5'
            ],
            'val': [
                'FloorPlan_Val1_1', 'FloorPlan_Val1_2', 'FloorPlan_Val1_3',
                'FloorPlan_Val2_1', 'FloorPlan_Val2_2', 'FloorPlan_Val2_3',
                'FloorPlan_Val3_1', 'FloorPlan_Val3_2', 'FloorPlan_Val3_3',
                'FloorPlan_Val4_1', 'FloorPlan_Val4_2', 'FloorPlan_Val4_3',
                'FloorPlan_Val5_1', 'FloorPlan_Val5_2', 'FloorPlan_Val5_3'
            ],
            'test': [
                'FloorPlan_Test1_1', 'FloorPlan_Test1_2',
                'FloorPlan_Test2_1', 'FloorPlan_Test2_2',
                'FloorPlan_Test3_1', 'FloorPlan_Test3_2',
                'FloorPlan_Test4_1', 'FloorPlan_Test4_2',
                'FloorPlan_Test5_1', 'FloorPlan_Test5_2',
                'FloorPlan_Test6_1', 'FloorPlan_Test6_2',
                'FloorPlan_Test7_1', 'FloorPlan_Test7_2'
            ]
        }
    
    def split_ithor_scenes(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[str]]:
        """Split iTHOR scenes maintaining balance across room types"""
        train_scenes = []
        val_scenes = []
        test_scenes = []
        
        for room_type, scenes in self.ithor_scenes.items():
            shuffled_scenes = scenes.copy()
            random.shuffle(shuffled_scenes)
            
            n_total = len(shuffled_scenes)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_scenes.extend(shuffled_scenes[:n_train])
            val_scenes.extend(shuffled_scenes[n_train:n_train + n_val])
            test_scenes.extend(shuffled_scenes[n_train + n_val:])
            
            print(f"iTHOR {room_type}: {n_train} train, {n_val} val, {n_total - n_train - n_val} test")
        
        return {
            'train': train_scenes,
            'val': val_scenes,
            'test': test_scenes
        }
    
    def get_combined_splits(self) -> Dict[str, Dict[str, List[str]]]:
        ithor_splits = self.split_ithor_scenes()
        
        combined_splits = {
            'train': {
                'ithor': ithor_splits['train'],
                'robothor': self.robothor_splits['train'],
                'combined': ithor_splits['train'] + self.robothor_splits['train']
            },
            'val': {
                'ithor': ithor_splits['val'],
                'robothor': self.robothor_splits['val'],
                'combined': ithor_splits['val'] + self.robothor_splits['val']
            },
            'test': {
                'ithor': ithor_splits['test'],
                'robothor': self.robothor_splits['test'],
                'combined': ithor_splits['test'] + self.robothor_splits['test']
            }
        }
        
        for split in ['train', 'val', 'test']:
            random.shuffle(combined_splits[split]['combined'])
        
        return combined_splits
    
    def save_combined_splits(self, output_dir: str = './config/splits'):
        os.makedirs(output_dir, exist_ok=True)
        
        combined_splits = self.get_combined_splits()
        
        for split_name in ['train', 'val', 'test']:
            filename = os.path.join(output_dir, f'combined_{split_name}_scenes.yaml')
            with open(filename, 'w') as f:
                yaml.dump({
                    'dataset': 'combined',
                    'split': split_name,
                    'scenes': combined_splits[split_name]['combined'],
                    'num_scenes': len(combined_splits[split_name]['combined']),
                    'num_ithor': len(combined_splits[split_name]['ithor']),
                    'num_robothor': len(combined_splits[split_name]['robothor']),
                    'ithor_scenes': combined_splits[split_name]['ithor'],
                    'robothor_scenes': combined_splits[split_name]['robothor']
                }, f, default_flow_style=False)
            
            print(f"Saved {split_name} split: {len(combined_splits[split_name]['combined'])} scenes "
                  f"(iTHOR: {len(combined_splits[split_name]['ithor'])}, "
                  f"RoboTHOR: {len(combined_splits[split_name]['robothor'])})")
        
        master_splits = {
            'train': combined_splits['train']['combined'],
            'val': combined_splits['val']['combined'],
            'test': combined_splits['test']['combined']
        }
        
        combined_file = os.path.join(output_dir, 'combined_splits.yaml')
        with open(combined_file, 'w') as f:
            yaml.dump(master_splits, f, default_flow_style=False)
        
        metadata_file = os.path.join(output_dir, 'combined_metadata.yaml')
        with open(metadata_file, 'w') as f:
            yaml.dump({
                'dataset': 'combined_ithor_robothor',
                'total_scenes': sum(len(split['combined']) for split in combined_splits.values()),
                'splits': {
                    split_name: {
                        'total': len(combined_splits[split_name]['combined']),
                        'ithor': len(combined_splits[split_name]['ithor']),
                        'robothor': len(combined_splits[split_name]['robothor'])
                    }
                    for split_name in ['train', 'val', 'test']
                },
                'ithor_room_types': list(self.ithor_scenes.keys()),
                'seed': 42
            }, f, default_flow_style=False)
        
        print(f"\nSaved combined splits to {combined_file}")
        print(f"Saved metadata to {metadata_file}")
        
        return combined_splits
    
    def print_statistics(self, splits: Dict[str, Dict[str, List[str]]]):
        print("\n" + "="*60)
        print("COMBINED DATASET STATISTICS")
        print("="*60)
        
        total_scenes = 0
        for split_name in ['train', 'val', 'test']:
            split_data = splits[split_name]
            total = len(split_data['combined'])
            ithor = len(split_data['ithor'])
            robothor = len(split_data['robothor'])
            total_scenes += total
            
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total: {total} scenes")
            print(f"  - iTHOR: {ithor} ({ithor/total*100:.1f}%)")
            print(f"  - RoboTHOR: {robothor} ({robothor/total*100:.1f}%)")
            
            room_counts = self._count_room_types(split_data['ithor'])
            if room_counts:
                print("  iTHOR room types:")
                for room_type, count in room_counts.items():
                    print(f"    - {room_type}: {count}")
        
        print(f"\nTotal scenes in dataset: {total_scenes}")
        print("="*60)
    
    def _count_room_types(self, scenes: List[str]) -> Dict[str, int]:
        room_counts = {'kitchen': 0, 'living_room': 0, 'bedroom': 0, 'bathroom': 0}
        
        for scene in scenes:
            if scene.startswith('FloorPlan'):
                try:
                    num = int(''.join(filter(str.isdigit, scene)))
                    if 1 <= num <= 30:
                        room_counts['kitchen'] += 1
                    elif 201 <= num <= 230:
                        room_counts['living_room'] += 1
                    elif 301 <= num <= 330:
                        room_counts['bedroom'] += 1
                    elif 401 <= num <= 430:
                        room_counts['bathroom'] += 1
                except:
                    pass
        
        return room_counts
    
    def verify_splits(self, splits: Dict[str, Dict[str, List[str]]]):
        all_scenes = set()
        
        for split_name in ['train', 'val', 'test']:
            scenes = set(splits[split_name]['combined'])
            
            overlap = all_scenes.intersection(scenes)
            if overlap:
                print(f"WARNING: {split_name} overlaps with other splits: {overlap}")
            
            all_scenes.update(scenes)
        
        print(f"\nVerification complete: {len(all_scenes)} unique scenes across all splits")

if __name__ == "__main__":
    splitter = CombinedAI2THORDatasetSplitter(seed=42)
    
    print("Generating combined iTHOR + RoboTHOR dataset splits...")
    combined_splits = splitter.save_combined_splits()
    
    splitter.print_statistics(combined_splits)
    splitter.verify_splits(combined_splits)