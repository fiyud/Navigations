import sys
sys.path.append('/home/tuandang/tuandang/quanganh/visualnav-transformer/train')

import ai2thor
from ai2thor.controller import Controller
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple
from tqdm import tqdm

class SceneGridCalculator:
    def __init__(self, grid_size: float = 0.25):
        self.grid_size = grid_size
        self.controller = None
        
    def compute_scene_grid_size(self, scene_name: str) -> int:
        """Compute the number of navigable grid cells in a scene"""
        try:
            # Initialize controller for the scene
            if self.controller is None:
                self.controller = Controller(
                    agentMode="locobot",
                    scene=scene_name,
                    gridSize=self.grid_size,
                    snapToGrid=True,
                    width=224,
                    height=224,
                    fieldOfView=90
                )
            else:
                self.controller.reset(scene=scene_name)
            
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]
            
            if not reachable_positions:
                print(f"Warning: No reachable positions found for {scene_name}")
                return 0
            
            # Calculate grid size from reachable positions
            x_coords = [pos['x'] for pos in reachable_positions]
            z_coords = [pos['z'] for pos in reachable_positions]
            
            # Method 1: Count unique grid cells
            unique_cells = set()
            for pos in reachable_positions:
                grid_x = int(pos['x'] / self.grid_size)
                grid_z = int(pos['z'] / self.grid_size)
                unique_cells.add((grid_x, grid_z))
            
            grid_count = len(unique_cells)
            
            # Method 2: Calculate approximate area
            x_range = max(x_coords) - min(x_coords)
            z_range = max(z_coords) - min(z_coords)
            area_estimate = (x_range * z_range) / (self.grid_size ** 2)
            
            print(f"{scene_name}: {grid_count} navigable grids "
                  f"(area: {x_range:.1f}x{z_range:.1f}m = {area_estimate:.0f} grids)")
            
            return grid_count
            
        except Exception as e:
            print(f"Error computing grid size for {scene_name}: {str(e)}")
            return 0
    
    def compute_all_scene_grids(self, scene_names: List[str]) -> Dict[str, int]:
        """Compute grid sizes for all scenes"""
        grid_sizes = {}
        
        print(f"Computing grid sizes for {len(scene_names)} scenes...")
        for scene_name in tqdm(scene_names):
            grid_size = self.compute_scene_grid_size(scene_name)
            grid_sizes[scene_name] = grid_size
        
        if self.controller:
            self.controller.stop()
        
        return grid_sizes
    
    def analyze_grid_distribution(self, grid_sizes: Dict[str, int]):
        print("\n=== Grid Size Distribution ===")
        
        # Separate by dataset type
        ithor_grids = {}
        robothor_grids = {}
        
        for scene, size in grid_sizes.items():
            if 'Train' in scene or 'Val' in scene or 'Test' in scene:
                robothor_grids[scene] = size
            else:
                ithor_grids[scene] = size
        
        # Analyze iTHOR by room type
        room_type_grids = {
            'kitchen': [],
            'living_room': [],
            'bedroom': [],
            'bathroom': []
        }
        
        for scene, size in ithor_grids.items():
            if scene.startswith('FloorPlan'):
                try:
                    num = int(''.join(filter(str.isdigit, scene)))
                    if 1 <= num <= 30:
                        room_type_grids['kitchen'].append(size)
                    elif 201 <= num <= 230:
                        room_type_grids['living_room'].append(size)
                    elif 301 <= num <= 330:
                        room_type_grids['bedroom'].append(size)
                    elif 401 <= num <= 430:
                        room_type_grids['bathroom'].append(size)
                except:
                    pass
        
        print("\niTHOR Room Types:")
        for room_type, sizes in room_type_grids.items():
            if sizes:
                print(f"  {room_type}: "
                      f"min={min(sizes)}, max={max(sizes)}, "
                      f"avg={np.mean(sizes):.1f}, count={len(sizes)}")
        
        if robothor_grids:
            robothor_sizes = list(robothor_grids.values())
            print(f"\nRoboTHOR: "
                  f"min={min(robothor_sizes)}, max={max(robothor_sizes)}, "
                  f"avg={np.mean(robothor_sizes):.1f}, count={len(robothor_sizes)}")
        
        # Overall statistics
        all_sizes = list(grid_sizes.values())
        print(f"\nOverall: "
              f"min={min(all_sizes)}, max={max(all_sizes)}, "
              f"avg={np.mean(all_sizes):.1f}, total={len(all_sizes)}")
        
        # Distribution percentiles
        percentiles = [10, 25, 50, 75, 90]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p}th: {np.percentile(all_sizes, p):.0f}")
        
        return room_type_grids

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compute grid sizes for AI2THOR scenes')
    parser.add_argument('--dataset', type=str, choices=['ithor', 'robothor', 'combined', 'sample'],
                       default='sample', help='Which dataset to process')
    parser.add_argument('--output-dir', type=str, default='./config/splits',
                       help='Directory to save grid sizes')
    args = parser.parse_args()
    
    if args.dataset == 'sample':
        scene_names = [
            'FloorPlan1', 'FloorPlan10', 'FloorPlan20',  # Kitchens
            'FloorPlan201', 'FloorPlan210',  # Living rooms
            'FloorPlan301', 'FloorPlan310',  # Bedrooms
            'FloorPlan401', 'FloorPlan410',  # Bathrooms
            'FloorPlan_Train1_1', 'FloorPlan_Train5_3',  # RoboTHOR
        ]
    else:
        # Load from splits file
        splits_file = os.path.join(args.output_dir, f'{args.dataset}_splits.yaml')
        if not os.path.exists(splits_file):
            splits_file = os.path.join(args.output_dir, 'combined_splits.yaml')
        
        if os.path.exists(splits_file):
            with open(splits_file, 'r') as f:
                splits = yaml.safe_load(f)
            scene_names = []
            for split_scenes in splits.values():
                scene_names.extend(split_scenes)
            scene_names = list(set(scene_names))  # Remove duplicates
        else:
            print(f"Splits file not found: {splits_file}")
            return
    
    # Compute grid sizes
    calculator = SceneGridCalculator(grid_size=0.25)
    grid_sizes = calculator.compute_all_scene_grids(scene_names)
    
    # Analyze distribution
    calculator.analyze_grid_distribution(grid_sizes)
    
    output_file = os.path.join(args.output_dir, '/home/tuandang/tuandang/quanganh/visualnav-transformer/train/ode/grid/grid.yaml')
    with open(output_file, 'w') as f:
        yaml.dump(grid_sizes, f, default_flow_style=False)
    
    print(f"\nSaved grid sizes to {output_file}")
    
    sorted_scenes = sorted(grid_sizes.items(), key=lambda x: x[1])
    
    print("\nSmallest scenes:")
    for scene, size in sorted_scenes[:5]:
        print(f"  {scene}: {size} grids")
    
    print("\nLargest scenes:")
    for scene, size in sorted_scenes[-5:]:
        print(f"  {scene}: {size} grids")

if __name__ == "__main__":
    main()