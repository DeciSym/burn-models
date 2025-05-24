#!/usr/bin/env python3
"""
Compare the detailed generation results from Python and Rust implementations.
"""

import json
import sys
from typing import Dict, List, Tuple

def load_results(filename: str) -> Dict:
    """Load generation results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def compare_tokens(python_results: Dict, rust_results: Dict) -> List[Dict]:
    """Compare token generation step by step."""
    comparisons = []
    
    # Compare initial tokens
    if python_results["initial_tokens"] != rust_results["initial_tokens"]:
        print("WARNING: Initial tokens differ!")
        print(f"Python: {python_results['initial_tokens']}")
        print(f"Rust: {rust_results['initial_tokens']}")
    
    # Compare each generation step
    for i, (py_step, rs_step) in enumerate(zip(python_results["steps"], rust_results["steps"])):
        comparison = {
            "step": i + 1,
            "matches": py_step["selected_token_id"] == rs_step["selected_token_id"],
            "python_token": {
                "id": py_step["selected_token_id"],
                "string": py_step["selected_token_string"],
                "prob": py_step["selected_token_prob"],
                "logit": py_step["selected_token_logit"]
            },
            "rust_token": {
                "id": rs_step["selected_token_id"],
                "string": rs_step["selected_token_string"],
                "prob": rs_step["selected_token_prob"],
                "logit": rs_step["selected_token_logit"]
            },
            "prob_diff": abs(py_step["selected_token_prob"] - rs_step["selected_token_prob"]),
            "logit_diff": abs(py_step["selected_token_logit"] - rs_step["selected_token_logit"]),
            "top_10_comparison": []
        }
        
        # Compare top 10 predictions
        py_top_ids = [pred["token_id"] for pred in py_step["top_10_predictions"]]
        rs_top_ids = [pred["token_id"] for pred in rs_step["top_10_predictions"]]
        
        for j in range(10):
            py_pred = py_step["top_10_predictions"][j]
            rs_pred = rs_step["top_10_predictions"][j]
            
            comparison["top_10_comparison"].append({
                "rank": j + 1,
                "matches": py_pred["token_id"] == rs_pred["token_id"],
                "python": {
                    "id": py_pred["token_id"],
                    "string": py_pred["token_string"],
                    "prob": py_pred["probability"]
                },
                "rust": {
                    "id": rs_pred["token_id"],
                    "string": rs_pred["token_string"],
                    "prob": rs_pred["probability"]
                },
                "prob_diff": abs(py_pred["probability"] - rs_pred["probability"])
            })
        
        # Calculate overlap in top 10
        overlap = len(set(py_top_ids) & set(rs_top_ids))
        comparison["top_10_overlap"] = overlap
        
        # Compare logits statistics
        py_stats = py_step["logits_stats"]
        rs_stats = rs_step["logits_stats"]
        comparison["logits_stats_diff"] = {
            "mean": abs(py_stats["mean"] - rs_stats["mean"]),
            "std": abs(py_stats["std"] - rs_stats["std"]),
            "min": abs(py_stats["min"] - rs_stats["min"]),
            "max": abs(py_stats["max"] - rs_stats["max"])
        }
        
        comparisons.append(comparison)
    
    return comparisons

def print_comparison_summary(comparisons: List[Dict]):
    """Print a summary of the comparison results."""
    print("\n" + "=" * 80)
    print("GENERATION COMPARISON SUMMARY")
    print("=" * 80)
    
    # Find where generation diverges
    divergence_point = None
    for i, comp in enumerate(comparisons):
        if not comp["matches"]:
            divergence_point = i + 1
            break
    
    if divergence_point:
        print(f"\n❌ Generation diverges at step {divergence_point}")
    else:
        print("\n✅ All generated tokens match!")
    
    # Print step-by-step comparison
    print("\nStep-by-step comparison:")
    for comp in comparisons:
        step = comp["step"]
        matches = "✓" if comp["matches"] else "✗"
        py_token = comp["python_token"]
        rs_token = comp["rust_token"]
        
        print(f"\nStep {step}: {matches}")
        print(f"  Python: '{py_token['string']}' (id: {py_token['id']}, prob: {py_token['prob']:.6f}, logit: {py_token['logit']:.4f})")
        print(f"  Rust:   '{rs_token['string']}' (id: {rs_token['id']}, prob: {rs_token['prob']:.6f}, logit: {rs_token['logit']:.4f})")
        print(f"  Prob diff: {comp['prob_diff']:.6f}, Logit diff: {comp['logit_diff']:.4f}")
        print(f"  Top-10 overlap: {comp['top_10_overlap']}/10")
        
        # Show top 5 predictions if they don't match
        if not comp["matches"]:
            print("  Top 5 predictions:")
            for j in range(5):
                top_comp = comp["top_10_comparison"][j]
                py = top_comp["python"]
                rs = top_comp["rust"]
                match_str = "✓" if top_comp["matches"] else "✗"
                print(f"    {j+1}. {match_str} Python: '{py['string']}' ({py['prob']:.4f}) | Rust: '{rs['string']}' ({rs['prob']:.4f})")
        
        # Show logits statistics differences
        stats_diff = comp["logits_stats_diff"]
        print(f"  Logits stats diff - Mean: {stats_diff['mean']:.4f}, Std: {stats_diff['std']:.4f}")
    
    # Calculate overall statistics
    total_steps = len(comparisons)
    matching_steps = sum(1 for comp in comparisons if comp["matches"])
    avg_prob_diff = sum(comp["prob_diff"] for comp in comparisons) / total_steps
    avg_logit_diff = sum(comp["logit_diff"] for comp in comparisons) / total_steps
    avg_top10_overlap = sum(comp["top_10_overlap"] for comp in comparisons) / total_steps
    
    print("\n" + "-" * 80)
    print("Overall Statistics:")
    print(f"  Matching tokens: {matching_steps}/{total_steps} ({matching_steps/total_steps*100:.1f}%)")
    print(f"  Average probability difference: {avg_prob_diff:.6f}")
    print(f"  Average logit difference: {avg_logit_diff:.4f}")
    print(f"  Average top-10 overlap: {avg_top10_overlap:.1f}/10")

def main():
    # Load results
    try:
        python_results = load_results("python_generation_details.json")
        rust_results = load_results("rust_generation_details.json")
    except FileNotFoundError as e:
        print(f"Error: Could not find results file - {e}")
        print("Please run both detailed_generation_comparison.py and the Rust equivalent first.")
        sys.exit(1)
    
    # Basic validation
    print("Loaded results:")
    print(f"  Python: {len(python_results['steps'])} steps")
    print(f"  Rust: {len(rust_results['steps'])} steps")
    
    if len(python_results['steps']) != len(rust_results['steps']):
        print("WARNING: Different number of generation steps!")
    
    # Compare results
    comparisons = compare_tokens(python_results, rust_results)
    
    # Print summary
    print_comparison_summary(comparisons)
    
    # Save detailed comparison
    output_file = "generation_comparison_analysis.json"
    with open(output_file, 'w') as f:
        json.dump({
            "comparisons": comparisons,
            "python_final_text": python_results["final_text"],
            "rust_final_text": rust_results["final_text"],
            "texts_match": python_results["final_text"] == rust_results["final_text"]
        }, f, indent=2)
    
    print(f"\nDetailed comparison saved to {output_file}")

if __name__ == "__main__":
    main()