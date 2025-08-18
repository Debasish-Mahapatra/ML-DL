"""
Enhanced domain adaptation script with physics-informed transfer learning.
"""

import os
import sys
import argparse
from pathlib import Path
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything

from src.utils.config import get_config
from src.data.data_loader import LightningDataModule
from src.training.trainer import create_domain_adaptation_trainer

# Enhanced imports for physics analysis
from src.utils.debug_utils import get_debug_manager, get_physics_monitor, debug_print
from src.utils.visualization import plot_cape_physics_analysis, plot_regime_performance_analysis

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_domain_cape_distributions(source_datamodule: LightningDataModule,
                                    target_datamodule: LightningDataModule,
                                    output_dir: str) -> dict:
    """
    Analyze CAPE distributions between source and target domains.
    This helps understand how physics should transfer.
    """
    
    logger.info("Analyzing CAPE distributions across domains...")
    
    # Physics thresholds
    thresholds = {
        'no_lightning': 1000.0,
        'moderate': 2500.0,
        'high': 4000.0,
        'saturation': 5000.0
    }
    
    def analyze_domain_data(datamodule, domain_name):
        """Analyze CAPE data for one domain."""
        cape_values = []
        lightning_values = []
        
        # Sample a few batches for analysis
        try:
            dataloader = datamodule.train_dataloader()
            for i, batch in enumerate(dataloader):
                if i >= 10:  # Analyze first 10 batches
                    break
                
                cape_data = batch['cape'].numpy()
                lightning_data = batch['lightning'].numpy()
                
                cape_values.extend(cape_data.flatten())
                lightning_values.extend(lightning_data.flatten())
        
        except Exception as e:
            logger.warning(f"Could not analyze {domain_name} data: {e}")
            return {}
        
        if not cape_values:
            return {}
        
        cape_array = np.array(cape_values)
        lightning_array = np.array(lightning_values)
        
        # Remove invalid values
        valid_mask = ~np.isnan(cape_array)
        cape_array = cape_array[valid_mask]
        lightning_array = lightning_array[valid_mask]
        
        # Calculate regime distributions
        regime_stats = {}
        for regime, threshold in thresholds.items():
            if regime == 'no_lightning':
                mask = cape_array < threshold
            elif regime == 'moderate':
                mask = (cape_array >= thresholds['no_lightning']) & (cape_array < threshold)
            elif regime == 'high':
                mask = (cape_array >= thresholds['moderate']) & (cape_array < threshold)
            else:  # very_high
                mask = cape_array >= thresholds['high']
            
            if mask.sum() > 0:
                regime_cape = cape_array[mask]
                regime_lightning = lightning_array[mask]
                
                regime_stats[regime] = {
                    'pixel_count': int(mask.sum()),
                    'percentage': float(mask.mean() * 100),
                    'cape_mean': float(regime_cape.mean()),
                    'cape_std': float(regime_cape.std()),
                    'lightning_rate': float(regime_lightning.mean())
                }
        
        return {
            'domain': domain_name,
            'cape_stats': {
                'min': float(cape_array.min()),
                'max': float(cape_array.max()),
                'mean': float(cape_array.mean()),
                'std': float(cape_array.std()),
                'total_pixels': len(cape_array)
            },
            'regime_stats': regime_stats,
            'lightning_overall_rate': float(lightning_array.mean())
        }
    
    # Analyze both domains
    source_analysis = analyze_domain_data(source_datamodule, "source")
    target_analysis = analyze_domain_data(target_datamodule, "target")
    
    # Compare domains
    comparison = {
        'source_analysis': source_analysis,
        'target_analysis': target_analysis,
        'domain_comparison': {}
    }
    
    if source_analysis and target_analysis:
        # Calculate domain differences
        source_cape = source_analysis['cape_stats']
        target_cape = target_analysis['cape_stats']
        
        comparison['domain_comparison'] = {
            'cape_mean_difference': target_cape['mean'] - source_cape['mean'],
            'cape_std_difference': target_cape['std'] - source_cape['std'],
            'lightning_rate_difference': target_analysis['lightning_overall_rate'] - source_analysis['lightning_overall_rate'],
            'regime_distribution_changes': {}
        }
        
        # Compare regime distributions
        for regime in thresholds.keys():
            if (regime in source_analysis['regime_stats'] and 
                regime in target_analysis['regime_stats']):
                
                source_pct = source_analysis['regime_stats'][regime]['percentage']
                target_pct = target_analysis['regime_stats'][regime]['percentage']
                
                comparison['domain_comparison']['regime_distribution_changes'][regime] = {
                    'source_percentage': source_pct,
                    'target_percentage': target_pct,
                    'percentage_change': target_pct - source_pct
                }
        
        # Log key findings
        logger.info(f"Domain Analysis Results:")
        logger.info(f"  Source CAPE: {source_cape['mean']:.1f}±{source_cape['std']:.1f} J/kg")
        logger.info(f"  Target CAPE: {target_cape['mean']:.1f}±{target_cape['std']:.1f} J/kg")
        logger.info(f"  CAPE difference: {comparison['domain_comparison']['cape_mean_difference']:.1f} J/kg")
        logger.info(f"  Lightning rate difference: {comparison['domain_comparison']['lightning_rate_difference']:.4f}")
    
    # Save analysis
    output_file = Path(output_dir) / "domain_cape_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    logger.info(f"Saved domain analysis to {output_file}")
    return comparison

def evaluate_physics_transfer(model, source_datamodule, target_datamodule, 
                            output_dir: str, step: str = "initial") -> dict:
    """
    Evaluate how well physics constraints transfer between domains.
    """
    
    logger.info(f"Evaluating physics transfer ({step})...")
    
    def evaluate_domain_physics(datamodule, domain_name):
        """Evaluate physics performance on one domain."""
        try:
            dataloader = datamodule.val_dataloader()
            
            all_predictions = []
            all_cape = []
            all_targets = []
            
            model.eval()
            device = next(model.parameters()).device
            
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= 5:  # Evaluate on first 5 batches
                        break
                    
                    cape_data = batch['cape'].to(device)
                    terrain_data = batch['terrain'].to(device)
                    lightning_targets = batch['lightning']
                    
                    # Get predictions
                    outputs = model(cape_data, terrain_data)
                    predictions = torch.sigmoid(outputs['lightning_prediction']).cpu()
                    
                    all_predictions.extend(predictions.flatten().numpy())
                    all_cape.extend(cape_data.cpu().flatten().numpy())
                    all_targets.extend(lightning_targets.flatten().numpy())
            
            if not all_predictions:
                return {}
            
            pred_array = np.array(all_predictions)
            cape_array = np.array(all_cape)
            target_array = np.array(all_targets)
            
            # Remove invalid values
            valid_mask = ~(np.isnan(cape_array) | np.isnan(pred_array))
            pred_array = pred_array[valid_mask]
            cape_array = cape_array[valid_mask]
            target_array = target_array[valid_mask]
            
            # Calculate physics metrics
            cape_pred_correlation = np.corrcoef(cape_array, pred_array)[0, 1] if len(cape_array) > 1 else 0.0
            cape_target_correlation = np.corrcoef(cape_array, target_array)[0, 1] if len(cape_array) > 1 else 0.0
            
            # Regime-specific analysis
            thresholds = {
                'no_lightning': 1000.0,
                'moderate': 2500.0,
                'high': 4000.0,
                'saturation': 5000.0
            }
            
            expected_rates = {
                'no_lightning': 0.05,
                'moderate': 0.25,
                'high': 0.50,
                'very_high': 0.75
            }
            
            regime_analysis = {}
            regimes = [
                ('no_lightning', cape_array < thresholds['no_lightning']),
                ('moderate', (cape_array >= thresholds['no_lightning']) & 
                            (cape_array < thresholds['moderate'])),
                ('high', (cape_array >= thresholds['moderate']) & 
                         (cape_array < thresholds['high'])),
                ('very_high', cape_array >= thresholds['high'])
            ]
            
            for regime_name, mask in regimes:
                if mask.sum() == 0:
                    continue
                
                regime_preds = pred_array[mask]
                actual_rate = (regime_preds > 0.5).mean()
                expected_rate = expected_rates[regime_name]
                
                alignment_score = 1.0 - abs(actual_rate - expected_rate)
                
                regime_analysis[regime_name] = {
                    'pixel_count': int(mask.sum()),
                    'actual_lightning_rate': float(actual_rate),
                    'expected_lightning_rate': float(expected_rate),
                    'physics_alignment': float(alignment_score)
                }
            
            return {
                'domain': domain_name,
                'cape_prediction_correlation': float(cape_pred_correlation) if not np.isnan(cape_pred_correlation) else 0.0,
                'cape_target_correlation': float(cape_target_correlation) if not np.isnan(cape_target_correlation) else 0.0,
                'regime_analysis': regime_analysis,
                'mean_prediction': float(pred_array.mean()),
                'prediction_std': float(pred_array.std())
            }
            
        except Exception as e:
            logger.warning(f"Could not evaluate {domain_name} physics: {e}")
            return {}
    
    # Evaluate both domains
    source_physics = evaluate_domain_physics(source_datamodule, "source")
    target_physics = evaluate_domain_physics(target_datamodule, "target")
    
    # Compare physics transfer
    transfer_analysis = {
        'step': step,
        'source_physics': source_physics,
        'target_physics': target_physics,
        'transfer_quality': {}
    }
    
    if source_physics and target_physics:
        # Calculate transfer quality metrics
        correlation_transfer = abs(target_physics['cape_prediction_correlation'] - 
                                 source_physics['cape_prediction_correlation'])
        
        # Average alignment across regimes
        def get_avg_alignment(physics_data):
            alignments = [data['physics_alignment'] 
                         for data in physics_data['regime_analysis'].values()]
            return np.mean(alignments) if alignments else 0.0
        
        source_avg_alignment = get_avg_alignment(source_physics)
        target_avg_alignment = get_avg_alignment(target_physics)
        alignment_transfer = abs(target_avg_alignment - source_avg_alignment)
        
        transfer_analysis['transfer_quality'] = {
            'correlation_difference': float(correlation_transfer),
            'alignment_difference': float(alignment_transfer),
            'source_correlation': source_physics['cape_prediction_correlation'],
            'target_correlation': target_physics['cape_prediction_correlation'],
            'source_avg_alignment': float(source_avg_alignment),
            'target_avg_alignment': float(target_avg_alignment),
            'transfer_success': float(correlation_transfer < 0.2 and alignment_transfer < 0.2)  # Good transfer if differences < 0.2
        }
        
        # Log transfer quality
        logger.info(f"Physics Transfer Quality ({step}):")
        logger.info(f"  Source correlation: {source_physics['cape_prediction_correlation']:.4f}")
        logger.info(f"  Target correlation: {target_physics['cape_prediction_correlation']:.4f}")
        logger.info(f"  Correlation difference: {correlation_transfer:.4f}")
        logger.info(f"  Average alignment difference: {alignment_transfer:.4f}")
        logger.info(f"  Transfer success: {'✅' if transfer_analysis['transfer_quality']['transfer_success'] > 0.5 else '❌'}")
    
    # Save transfer analysis
    output_file = Path(output_dir) / f"physics_transfer_analysis_{step}.json"
    with open(output_file, 'w') as f:
        json.dump(transfer_analysis, f, indent=2, default=str)
    
    return transfer_analysis

def create_domain_comparison_plots(source_datamodule, target_datamodule, 
                                 model, output_dir: str):
    """Create visual comparisons between source and target domains."""
    
    logger.info("Creating domain comparison visualizations...")
    
    try:
        # Get sample data from each domain
        def get_sample_data(datamodule, domain_name):
            try:
                dataloader = datamodule.val_dataloader()
                batch = next(iter(dataloader))
                
                cape_data = batch['cape'][0, 0].numpy()  # First sample, remove batch and channel dims
                lightning_data = batch['lightning'][0].numpy()  # First sample
                
                return cape_data, lightning_data
            except Exception as e:
                logger.warning(f"Could not get sample data for {domain_name}: {e}")
                return None, None
        
        source_cape, source_lightning = get_sample_data(source_datamodule, "source")
        target_cape, target_lightning = get_sample_data(target_datamodule, "target")
        
        if source_cape is not None and target_cape is not None:
            # Create comparison plots
            
            # Source domain physics
            plot_cape_physics_analysis(
                source_cape, source_lightning, source_lightning,
                output_file=f"{output_dir}/source_domain_physics.png",
                title="Source Domain (Odisha) - CAPE Physics Analysis"
            )
            
            # Target domain physics
            plot_cape_physics_analysis(
                target_cape, target_lightning, target_lightning,
                output_file=f"{output_dir}/target_domain_physics.png", 
                title="Target Domain - CAPE Physics Analysis"
            )
            
            logger.info("Created domain comparison plots")
        
    except Exception as e:
        logger.warning(f"Could not create domain comparison plots: {e}")

def main():
    """Enhanced domain adaptation with physics analysis."""
    
    parser = argparse.ArgumentParser(description="Enhanced Domain Adaptation for Lightning Prediction")
    parser.add_argument("--source-checkpoint", type=str, required=True,
                       help="Path to source domain checkpoint (Odisha)")
    parser.add_argument("--target-config", type=str, required=True,
                       help="Configuration for target domain data")
    parser.add_argument("--source-config", type=str, default=None,
                       help="Configuration for source domain data (for analysis)")
    parser.add_argument("--experiment-name", type=str,
                       default=f"domain_adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Experiment name")
    parser.add_argument("--max-epochs", type=int, default=20,
                       help="Maximum training epochs")
    parser.add_argument("--freeze-epochs", type=int, default=5,
                       help="Epochs to freeze backbone")
    parser.add_argument("--adaptation-lr", type=float, default=0.0001,
                       help="Learning rate for adaptation layers")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Enhanced physics analysis arguments
    parser.add_argument("--analyze-physics", action="store_true", default=True,
                       help="Perform physics transfer analysis (enabled by default)")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create domain comparison plots")
    parser.add_argument("--physics-debug", action="store_true",
                       help="Enable detailed physics debugging")
    
    args = parser.parse_args()
    
    # Set random seed
    seed_everything(args.seed, workers=True)
    
    # Create experiment directory
    experiment_dir = Path(f"experiments/{args.experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate source checkpoint
    if not Path(args.source_checkpoint).exists():
        logger.error(f"Source checkpoint not found: {args.source_checkpoint}")
        return 1
    
    # Load target domain configuration
    try:
        target_config = get_config(args.target_config)
        
        # Override domain adaptation settings
        if not hasattr(target_config.training, 'domain_adaptation'):
            target_config.training.domain_adaptation = {}
        
        target_config.training.domain_adaptation.enabled = True
        target_config.training.domain_adaptation.freeze_epochs = args.freeze_epochs
        target_config.training.domain_adaptation.lr_multiplier = 0.1
        target_config.training.max_epochs = args.max_epochs
        
        # Enable physics debugging if requested
        if args.physics_debug:
            if not hasattr(target_config.training, 'debug'):
                target_config.training.debug = {}
            target_config.training.debug.enabled = True
            target_config.training.debug.physics_debug = True
            target_config.training.debug.cape_analysis = True
        
        logger.info(f"Loaded target domain configuration")
        
    except Exception as e:
        logger.error(f"Failed to load target configuration: {e}")
        return 1
    
    # Initialize debug manager
    debug_manager = get_debug_manager(target_config)
    physics_monitor = get_physics_monitor(target_config)
    
    if debug_manager.physics_debug:
        debug_print("Enhanced domain adaptation with physics analysis enabled", "physics")
    
    # Setup target domain data
    try:
        logger.info("Setting up target domain data...")
        target_datamodule = LightningDataModule(target_config)
        target_datamodule.setup("fit")
        
        logger.info(f"Target domain data: {len(target_datamodule.train_files)} train files")
        
    except Exception as e:
        logger.error(f"Failed to setup target domain data: {e}")
        return 1
    
    # Setup source domain data for comparison (optional)
    source_datamodule = None
    if args.source_config and args.analyze_physics:
        try:
            logger.info("Setting up source domain data for comparison...")
            source_config = get_config(args.source_config)
            source_datamodule = LightningDataModule(source_config)
            source_datamodule.setup("fit")
            logger.info("Source domain data loaded for analysis")
        except Exception as e:
            logger.warning(f"Could not load source domain data: {e}")
    
    # Analyze domain CAPE distributions
    domain_analysis = None
    if args.analyze_physics and source_datamodule:
        try:
            domain_analysis = analyze_domain_cape_distributions(
                source_datamodule, target_datamodule, experiment_dir
            )
        except Exception as e:
            logger.warning(f"Domain analysis failed: {e}")
    
    # Create domain adaptation trainer
    try:
        logger.info("Creating domain adaptation trainer...")
        trainer, lightning_module = create_domain_adaptation_trainer(
            target_config,
            args.source_checkpoint,
            args.experiment_name
        )
        
        logger.info("Domain adaptation trainer created")
        
    except Exception as e:
        logger.error(f"Failed to create domain adaptation trainer: {e}")
        return 1
    
    # Initial physics evaluation (before adaptation)
    initial_physics = None
    if args.analyze_physics and source_datamodule:
        try:
            initial_physics = evaluate_physics_transfer(
                lightning_module, source_datamodule, target_datamodule,
                experiment_dir, "initial"
            )
        except Exception as e:
            logger.warning(f"Initial physics evaluation failed: {e}")
    
    # Create domain comparison plots
    if args.create_plots and source_datamodule:
        try:
            create_domain_comparison_plots(
                source_datamodule, target_datamodule, lightning_module, experiment_dir
            )
        except Exception as e:
            logger.warning(f"Plot creation failed: {e}")
    
    # Run domain adaptation training
    try:
        logger.info("Starting enhanced domain adaptation training...")
        trainer.fit(lightning_module, target_datamodule)
        
        logger.info("Domain adaptation completed")
        
    except Exception as e:
        logger.error(f"Domain adaptation training failed: {e}")
        return 1
    
    # Final physics evaluation (after adaptation)
    final_physics = None
    if args.analyze_physics and source_datamodule:
        try:
            final_physics = evaluate_physics_transfer(
                lightning_module, source_datamodule, target_datamodule,
                experiment_dir, "final"
            )
        except Exception as e:
            logger.warning(f"Final physics evaluation failed: {e}")
    
    # Test on target domain
    test_results = None
    if hasattr(target_datamodule, 'test_files') and target_datamodule.test_files:
        try:
            logger.info("Evaluating on target domain test set...")
            test_results = trainer.test(lightning_module, target_datamodule)
            logger.info(f"Target domain test results: {test_results}")
        except Exception as e:
            logger.warning(f"Target domain testing failed: {e}")
    
    # Create comprehensive transfer report
    try:
        transfer_report = {
            'experiment_name': args.experiment_name,
            'completion_time': datetime.now().isoformat(),
            'source_checkpoint': args.source_checkpoint,
            'target_config': args.target_config,
            'training_params': {
                'max_epochs': args.max_epochs,
                'freeze_epochs': args.freeze_epochs,
                'adaptation_lr': args.adaptation_lr
            },
            'domain_analysis': domain_analysis,
            'initial_physics': initial_physics,
            'final_physics': final_physics,
            'test_results': test_results
        }
        
        # Physics transfer summary
        if initial_physics and final_physics:
            initial_quality = initial_physics.get('transfer_quality', {})
            final_quality = final_physics.get('transfer_quality', {})
            
            transfer_report['physics_transfer_summary'] = {
                'initial_target_correlation': initial_quality.get('target_correlation', 0),
                'final_target_correlation': final_quality.get('target_correlation', 0),
                'correlation_improvement': (
                    final_quality.get('target_correlation', 0) - 
                    initial_quality.get('target_correlation', 0)
                ),
                'transfer_success': final_quality.get('transfer_success', 0)
            }
        
        # Save comprehensive report
        report_file = experiment_dir / "domain_adaptation_report.json"
        with open(report_file, 'w') as f:
            json.dump(transfer_report, f, indent=2, default=str)
        
        # Create summary markdown report
        markdown_content = f"""# Domain Adaptation Report

**Experiment**: {args.experiment_name}  
**Completed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration
- **Source Checkpoint**: {args.source_checkpoint}
- **Target Domain Config**: {args.target_config}
- **Training Epochs**: {args.max_epochs}
- **Freeze Epochs**: {args.freeze_epochs}

## Physics Transfer Results
"""
        
        if 'physics_transfer_summary' in transfer_report:
            pts = transfer_report['physics_transfer_summary']
            markdown_content += f"""
### CAPE-Prediction Correlation
- **Initial**: {pts['initial_target_correlation']:.4f}
- **Final**: {pts['final_target_correlation']:.4f}  
- **Improvement**: {pts['correlation_improvement']:.4f}
- **Transfer Success**: {'✅ Success' if pts['transfer_success'] > 0.5 else '❌ Needs Improvement'}

### Interpretation
- Correlation > 0.6: Excellent physics transfer
- Correlation 0.3-0.6: Good physics transfer  
- Correlation < 0.3: Poor physics transfer
"""
        
        if test_results:
            markdown_content += f"""
## Target Domain Performance
- **Test Results**: {test_results}
"""
        
        markdown_content += f"""
## Files Generated
- `domain_adaptation_report.json` - Complete technical report
- `domain_cape_analysis.json` - CAPE distribution analysis
- `physics_transfer_analysis_*.json` - Physics transfer evaluations
"""
        
        if args.create_plots:
            markdown_content += """- `*_domain_physics.png` - Domain comparison plots
"""
        
        summary_file = experiment_dir / "README.md"
        with open(summary_file, 'w') as f:
            f.write(markdown_content)
        
        logger.info(f"Saved comprehensive reports to {experiment_dir}")
        
    except Exception as e:
        logger.warning(f"Failed to create transfer report: {e}")
    
    # Final summary
    logger.info("=== DOMAIN ADAPTATION SUMMARY ===")
    logger.info(f"✅ Domain adaptation experiment '{args.experiment_name}' completed")
    
    if initial_physics and final_physics:
        pts = transfer_report.get('physics_transfer_summary', {})
        improvement = pts.get('correlation_improvement', 0)
        final_corr = pts.get('final_target_correlation', 0)
        
        logger.info(f"🔬 Physics Transfer Performance:")
        logger.info(f"   Final CAPE correlation: {final_corr:.4f}")
        logger.info(f"   Correlation improvement: {improvement:.4f}")
        
        if final_corr > 0.6:
            logger.info("🎯 EXCELLENT physics transfer achieved!")
        elif final_corr > 0.3:
            logger.info("✅ GOOD physics transfer achieved")
        else:
            logger.info("⚠️ Physics transfer needs improvement")
    
    if test_results:
        logger.info(f"📊 Target domain test performance: {test_results}")
    
    logger.info(f"📁 Results saved to: experiments/{args.experiment_name}/")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)