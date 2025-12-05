"""
YOLO Dataset Validation & Cleaning Script
==========================================
This script will:
1. Validate all YOLO label files
2. Fix out-of-bounds coordinates (clip to [0,1])
3. Remove invalid/empty annotations
4. Generate a detailed report
5. Create backups before making changes
6. Clear corrupted cache files
"""

import os
import glob
import shutil
import json
from datetime import datetime
from pathlib import Path

class YOLODatasetCleaner:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.detection_dir = os.path.join(dataset_dir, "detection")
        self.labels_dir = os.path.join(self.detection_dir, "labels")
        self.images_dir = os.path.join(self.detection_dir, "images")
        self.backup_dir = os.path.join(dataset_dir, f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        self.stats = {
            'total_files': 0,
            'corrupt_files': 0,
            'fixed_files': 0,
            'removed_files': 0,
            'empty_files': 0,
            'splits': {'train': {}, 'val': {}, 'test': {}}
        }
        
    def create_backup(self):
        """Create backup of labels directory"""
        print("\n" + "="*60)
        print("📦 CREATING BACKUP")
        print("="*60)
        
        if os.path.exists(self.backup_dir):
            print(f"⚠️  Backup directory already exists: {self.backup_dir}")
            return False
            
        try:
            shutil.copytree(self.labels_dir, os.path.join(self.backup_dir, "labels"))
            print(f"✅ Backup created at: {self.backup_dir}")
            return True
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return False
    
    def validate_label_line(self, line):
        """Validate a single label line and return fixed version if needed"""
        parts = line.strip().split()
        
        # Need at least 5 values: class_id x_center y_center width height
        if len(parts) < 5:
            return None, "insufficient_values"
        
        try:
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:5]]
            
            # Check if coordinates are out of bounds
            issues = []
            if any(c < 0 for c in coords):
                issues.append("negative_coords")
            if any(c > 1 for c in coords):
                issues.append("out_of_bounds")
            
            # Check for invalid dimensions (width/height must be > 0)
            if coords[2] <= 0 or coords[3] <= 0:
                issues.append("invalid_dimensions")
            
            # Clip coordinates to [0, 1]
            fixed_coords = [max(0.0, min(1.0, c)) for c in coords]
            
            # Keep width and height positive
            if fixed_coords[2] <= 0:
                fixed_coords[2] = 0.01
            if fixed_coords[3] <= 0:
                fixed_coords[3] = 0.01
            
            fixed_line = f"{class_id} {' '.join(f'{c:.6f}' for c in fixed_coords)}\n"
            
            return fixed_line, issues if issues else None
            
        except (ValueError, IndexError) as e:
            return None, f"parse_error: {e}"
    
    def clean_label_file(self, label_path):
        """Clean a single label file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            if not lines:
                return {'status': 'empty', 'issues': ['empty_file']}
            
            fixed_lines = []
            all_issues = []
            
            for line_num, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                    
                fixed_line, issues = self.validate_label_line(line)
                
                if fixed_line is None:
                    all_issues.append(f"line_{line_num}: {issues}")
                else:
                    fixed_lines.append(fixed_line)
                    if issues:
                        all_issues.extend([f"line_{line_num}: {issue}" for issue in issues])
            
            if not fixed_lines:
                return {'status': 'all_invalid', 'issues': all_issues}
            
            # Write back fixed labels
            with open(label_path, 'w') as f:
                f.writelines(fixed_lines)
            
            if all_issues:
                return {'status': 'fixed', 'issues': all_issues, 'lines_kept': len(fixed_lines)}
            else:
                return {'status': 'valid', 'issues': [], 'lines_kept': len(fixed_lines)}
                
        except Exception as e:
            return {'status': 'error', 'issues': [str(e)]}
    
    def process_split(self, split_name):
        """Process all label files in a split (train/val/test)"""
        print(f"\n📂 Processing {split_name.upper()} split...")
        
        label_path = os.path.join(self.labels_dir, split_name)
        image_path = os.path.join(self.images_dir, split_name)
        
        if not os.path.exists(label_path):
            print(f"⚠️  Labels directory not found: {label_path}")
            return
        
        label_files = glob.glob(os.path.join(label_path, "*.txt"))
        
        split_stats = {
            'total': len(label_files),
            'valid': 0,
            'fixed': 0,
            'empty': 0,
            'removed': 0,
            'corrupt_files': []
        }
        
        for label_file in label_files:
            self.stats['total_files'] += 1
            result = self.clean_label_file(label_file)
            
            if result['status'] == 'valid':
                split_stats['valid'] += 1
                
            elif result['status'] == 'fixed':
                split_stats['fixed'] += 1
                self.stats['fixed_files'] += 1
                split_stats['corrupt_files'].append({
                    'file': os.path.basename(label_file),
                    'issues': result['issues']
                })
                
            elif result['status'] in ['empty', 'all_invalid']:
                split_stats['empty'] += 1
                self.stats['empty_files'] += 1
                split_stats['corrupt_files'].append({
                    'file': os.path.basename(label_file),
                    'issues': result['issues']
                })
                
                # Remove empty/invalid label files and corresponding images
                img_file = label_file.replace(label_path, image_path).replace('.txt', '.jpg')
                try:
                    os.remove(label_file)
                    if os.path.exists(img_file):
                        os.remove(img_file)
                    split_stats['removed'] += 1
                    self.stats['removed_files'] += 1
                    print(f"  🗑️  Removed: {os.path.basename(label_file)}")
                except Exception as e:
                    print(f"  ❌ Could not remove {os.path.basename(label_file)}: {e}")
        
        self.stats['splits'][split_name] = split_stats
        
        print(f"  ✅ Valid: {split_stats['valid']}")
        print(f"  🔧 Fixed: {split_stats['fixed']}")
        print(f"  🗑️  Removed: {split_stats['removed']}")
    
    def clear_cache_files(self):
        """Remove YOLO cache files"""
        print("\n" + "="*60)
        print("🧹 CLEARING CACHE FILES")
        print("="*60)
        
        cache_files = glob.glob(os.path.join(self.labels_dir, "**/*.cache"), recursive=True)
        
        for cache_file in cache_files:
            try:
                os.remove(cache_file)
                print(f"  ✅ Removed: {cache_file}")
            except Exception as e:
                print(f"  ❌ Could not remove {cache_file}: {e}")
        
        print(f"✅ Removed {len(cache_files)} cache files")
    
    def generate_report(self):
        """Generate detailed cleaning report"""
        print("\n" + "="*60)
        print("📊 CLEANING REPORT")
        print("="*60)
        
        print(f"\n📈 Overall Statistics:")
        print(f"  Total files processed: {self.stats['total_files']}")
        print(f"  Files fixed: {self.stats['fixed_files']}")
        print(f"  Files removed: {self.stats['removed_files']}")
        print(f"  Empty files: {self.stats['empty_files']}")
        
        print(f"\n📊 Per-Split Statistics:")
        for split, data in self.stats['splits'].items():
            if data:
                print(f"\n  {split.upper()}:")
                print(f"    Total: {data['total']}")
                print(f"    Valid: {data['valid']}")
                print(f"    Fixed: {data['fixed']}")
                print(f"    Removed: {data['removed']}")
        
        # Save detailed report to JSON
        report_path = os.path.join(self.dataset_dir, f"cleaning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_path}")
    
    def verify_dataset(self):
        """Verify dataset after cleaning"""
        print("\n" + "="*60)
        print("✅ VERIFICATION")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            label_path = os.path.join(self.labels_dir, split)
            image_path = os.path.join(self.images_dir, split)
            
            label_files = glob.glob(os.path.join(label_path, "*.txt"))
            image_files = glob.glob(os.path.join(image_path, "*.jpg"))
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            
            if len(image_files) != len(label_files):
                print(f"  ⚠️  WARNING: Image/Label count mismatch!")
    
    def run(self):
        """Run the complete cleaning pipeline"""
        print("\n" + "="*60)
        print("🚀 YOLO DATASET CLEANER")
        print("="*60)
        print(f"Dataset directory: {self.dataset_dir}")
        
        # Step 1: Create backup
        if not self.create_backup():
            response = input("\n⚠️  Proceed without backup? (yes/no): ")
            if response.lower() != 'yes':
                print("❌ Cleaning cancelled.")
                return
        
        # Step 2: Process each split
        print("\n" + "="*60)
        print("🔧 CLEANING LABELS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            self.process_split(split)
        
        # Step 3: Clear cache
        self.clear_cache_files()
        
        # Step 4: Generate report
        self.generate_report()
        
        # Step 5: Verify
        self.verify_dataset()
        
        print("\n" + "="*60)
        print("✅ CLEANING COMPLETE!")
        print("="*60)
        print("\n🎯 Next Steps:")
        print("  1. Review the cleaning report")
        print("  2. Delete old training runs: rm -rf yolo_runs/smartvision_yolov8s*")
        print("  3. Retrain your model: python scripts/train_yolo_smartvision.py")
        print(f"\n💾 Backup location: {self.backup_dir}")
        print("   (You can restore from backup if needed)")


if __name__ == "__main__":
    # Configuration
    DATASET_DIR = "smartvision_dataset"
    
    # Run the cleaner
    cleaner = YOLODatasetCleaner(DATASET_DIR)
    cleaner.run()