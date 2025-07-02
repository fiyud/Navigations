def _load_splits(self, config: Dict) -> Dict[str, List[str]]:
    """Load dataset splits with comprehensive error handling"""
    dataset = config['dataset']
    
    # List of possible split file locations
    possible_files = [
        config.get('splits_file', f'./config/splits/{dataset}_splits.yaml'),
        f'./config/splits/{dataset}_splits.yaml',
        f'./config/splits/combined_splits.yaml',
        f'./train/ode/grid/gridConfig/{dataset}.yaml',
        f'./splits/{dataset}_splits.yaml'
    ]
    
    print(f"Loading splits for dataset: {dataset}")
    
    # Try each possible file location
    for splits_file in possible_files:
        if os.path.exists(splits_file):
            try:
                print(f"Attempting to load: {splits_file}")
                with open(splits_file, 'r') as f:
                    splits = yaml.safe_load(f)
                
                # Check if splits has the required structure
                if isinstance(splits, dict) and all(key in splits for key in ['train', 'val', 'test']):
                    # Verify splits contain actual scenes
                    total_scenes = sum(len(split) for split in splits.values())
                    if total_scenes > 0:
                        print(f"✓ Successfully loaded {total_scenes} scenes from {splits_file}")
                        for split, scenes in splits.items():
                            print(f"    {split}: {len(scenes)} scenes")
                        return splits
                    else:
                        print(f" File {splits_file} exists but contains no scenes")
                else:
                    print(f" File {splits_file} has invalid format")
                    
            except Exception as e:
                print(f" Failed to load {splits_file}: {e}")
                continue
    
    # If no valid file found, generate default splits
    print(" No valid splits file found. Generating default splits...")
    return self._generate_default_splits(dataset)