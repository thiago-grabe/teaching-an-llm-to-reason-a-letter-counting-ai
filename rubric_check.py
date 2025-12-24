#!/usr/bin/env python3
"""
Rubric Validation Script for Teaching an LLM to Reason Project
This script validates that all rubric requirements are met.

Rubric Criteria:
1. Model Setup - Valid LoRA config (rank in {8,16,32,64,128}, target modules)
2. Baseline Prompting - CoT prompt with at least one example
3. Reward Design & Validation - 5 reward functions with positive/negative rewards
4. Training & Monitoring - Longer training run with max_steps > 5
5. Final Comparison - Both models shown on at least one example
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Any


class Colors:
    """Terminal colors for output formatting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def load_notebook(notebook_path: str) -> Dict[str, Any]:
    """Load and parse the Jupyter notebook."""
    with open(notebook_path, 'r') as f:
        return json.load(f)


def get_all_code_source(notebook: Dict[str, Any]) -> str:
    """Extract all code cell sources as a single string."""
    sources = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                sources.append("".join(source))
            else:
                sources.append(source)
    return "\n".join(sources)


def check_model_setup(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check Model Setup: Valid LoRA config.

    Requirements:
    - lora_rank is one of {8, 16, 32, 64, 128}
    - target_modules includes q_proj, k_proj, v_proj, o_proj and/or gate_proj, up_proj, down_proj
    - Model instantiates without errors (checked by presence of LoraConfig)
    """
    results = {
        "passed": False,
        "lora_rank_valid": False,
        "lora_rank_value": None,
        "target_modules_valid": False,
        "target_modules": [],
        "has_lora_config": False,
        "details": []
    }

    valid_ranks = {8, 16, 32, 64, 128}
    required_attention_modules = {"q_proj", "k_proj", "v_proj", "o_proj"}
    optional_mlp_modules = {"gate_proj", "up_proj", "down_proj"}

    all_source = get_all_code_source(notebook)

    # Check for lora_rank assignment
    rank_match = re.search(r"lora_rank\s*=\s*(\d+)", all_source)
    if rank_match:
        rank = int(rank_match.group(1))
        results["lora_rank_value"] = rank
        if rank in valid_ranks:
            results["lora_rank_valid"] = True
            results["details"].append(f"lora_rank = {rank} (valid)")
        else:
            results["details"].append(f"lora_rank = {rank} NOT in {valid_ranks}")
    else:
        results["details"].append("lora_rank not found")

    # Check for target_modules
    modules_match = re.search(r"target_modules\s*=\s*\[(.*?)\]", all_source, re.DOTALL)
    if modules_match:
        modules_str = modules_match.group(1)
        modules = set(re.findall(r'"(\w+)"', modules_str))
        results["target_modules"] = list(modules)

        # Check if attention modules are included
        attention_included = required_attention_modules.intersection(modules)
        mlp_included = optional_mlp_modules.intersection(modules)

        if len(attention_included) >= 4 or (len(attention_included) >= 2 and len(mlp_included) >= 1):
            results["target_modules_valid"] = True
            results["details"].append(f"target_modules include: {sorted(modules)}")
        else:
            results["details"].append(f"Insufficient modules. Found: {sorted(modules)}")
    else:
        results["details"].append("target_modules not found")

    # Check for LoraConfig usage
    if "LoraConfig" in all_source or "get_peft_model" in all_source:
        results["has_lora_config"] = True
        results["details"].append("LoraConfig/PEFT configuration found")

    results["passed"] = (
        results["lora_rank_valid"] and
        results["target_modules_valid"] and
        results["has_lora_config"]
    )

    return results


def check_baseline_prompting(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check Baseline Prompting: CoT prompt with at least one example.

    Requirements:
    - SYSTEM_PROMPT with Chain-of-Thought elements
    - At least one concrete input-output example (few-shot)
    - Step-by-step reasoning visible
    """
    results = {
        "passed": False,
        "has_system_prompt": False,
        "has_cot_elements": False,
        "has_example": False,
        "has_numbered_steps": False,
        "details": []
    }

    all_source = get_all_code_source(notebook)

    # Look for SYSTEM_PROMPT with substantial content
    # Handle escaped newlines from notebook JSON format
    all_source_unescaped = all_source.replace('\\n', '\n')

    prompt_patterns = [
        r'SYSTEM_PROMPT\s*=\s*"""(.*?)"""',
        r"SYSTEM_PROMPT\s*=\s*'''(.*?)'''",
        r'SYSTEM_PROMPT\s*=\s*"([^"]{50,})"',
    ]

    prompt_content = ""
    for pattern in prompt_patterns:
        # Find ALL matches and use the longest one with CoT elements
        matches = re.findall(pattern, all_source_unescaped, re.DOTALL)
        for match in matches:
            if len(match) > len(prompt_content) and len(match) > 50:
                prompt_content = match

    if len(prompt_content) > 50:
        results["has_system_prompt"] = True
        results["details"].append(f"Found SYSTEM_PROMPT ({len(prompt_content)} chars)")

    if prompt_content:
        # Check for CoT indicators
        cot_indicators = ["step", "letter by letter", "reasoning", "count", "one by one", "spell"]
        cot_found = [ind for ind in cot_indicators if ind.lower() in prompt_content.lower()]
        if cot_found:
            results["has_cot_elements"] = True
            results["details"].append(f"CoT elements found: {cot_found[:3]}")

        # Check for example with numbered steps (e.g., "1. r - 0 o's so far")
        if re.search(r"\d+\.\s*[a-z]\s*-", prompt_content, re.IGNORECASE):
            results["has_example"] = True
            results["has_numbered_steps"] = True
            results["details"].append("Found numbered step-by-step example")

        # Check for <reasoning>/<answer> tags in example
        if "<reasoning>" in prompt_content and "<answer>" in prompt_content:
            results["details"].append("Found reasoning/answer format in example")
    else:
        results["details"].append("SYSTEM_PROMPT not found or too short")

    results["passed"] = (
        results["has_system_prompt"] and
        results["has_cot_elements"] and
        results["has_example"]
    )

    return results


def check_reward_design(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check Reward Design & Validation.

    Requirements:
    - 5 reward functions: numbering, spelling, counting, formatting, correctness
    - Each has positive rewards for correct behavior
    - Each has negative rewards for incorrect behavior
    - Validation at end shows higher reward for correct sample
    """
    results = {
        "passed": False,
        "functions_found": [],
        "functions_with_rewards": [],
        "has_validations": False,
        "details": []
    }

    required_functions = [
        "numbering_reward_func",
        "spelling_reward_func",
        "counting_reward_func",
        "format_reward_func",
        "correct_answer_reward_func"
    ]

    all_source = get_all_code_source(notebook)

    for func_name in required_functions:
        # Check if function is defined
        func_def_pattern = rf"def\s+{func_name}\s*\("
        if re.search(func_def_pattern, all_source):
            results["functions_found"].append(func_name)

            # Extract the function body (simplified)
            func_pattern = rf"def\s+{func_name}.*?(?=\ndef\s+\w+|\Z)"
            func_match = re.search(func_pattern, all_source, re.DOTALL)

            if func_match:
                func_body = func_match.group(0)

                # Check for reward values (positive and negative)
                has_positive = bool(re.search(r"reward\s*\+=\s*[\d.]+|[\d.]+\s+if.*else", func_body))
                has_negative = bool(re.search(r"reward\s*-=\s*[\d.]+|-[\d.]+", func_body))

                if has_positive or has_negative:
                    results["functions_with_rewards"].append(func_name)
                    results["details"].append(f"{func_name}: Has reward logic")
                else:
                    results["details"].append(f"{func_name}: Missing reward values")
        else:
            results["details"].append(f"{func_name}: NOT FOUND")

    # Check for validation assertions
    if "assert res[1] > res[0]" in all_source:
        results["has_validations"] = True
        results["details"].append("Found reward validation assertions")

    # Check REWARD_FUNCS list
    if "REWARD_FUNCS" in all_source and len(results["functions_found"]) >= 4:
        results["details"].append("REWARD_FUNCS list configured")

    results["passed"] = (
        len(results["functions_found"]) >= 5 and
        len(results["functions_with_rewards"]) >= 5 and
        results["has_validations"]
    )

    return results


def check_training_monitoring(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check Training & Monitoring.

    Requirements:
    - Longer training run (max_steps > 5)
    - GRPOConfig with valid parameters
    - Training should show increasing reward trend
    """
    results = {
        "passed": False,
        "has_quick_train": False,
        "has_longer_train": False,
        "max_steps_value": None,
        "has_grpo_config": False,
        "has_learning_rate": False,
        "has_plot": False,
        "details": []
    }

    all_source = get_all_code_source(notebook)

    # Check for GRPOConfig
    if "GRPOConfig" in all_source:
        results["has_grpo_config"] = True
        results["details"].append("GRPOConfig found")

    # Check for learning_rate setting
    lr_match = re.search(r"learning_rate\s*=\s*([\d.e\-]+)", all_source)
    if lr_match:
        results["has_learning_rate"] = True
        results["details"].append(f"learning_rate = {lr_match.group(1)}")

    # Find all max_steps values
    steps_matches = re.findall(r"max_steps\s*=\s*(\d+)", all_source)
    if steps_matches:
        steps_values = [int(s) for s in steps_matches]

        # Check for quick train (5 steps)
        if 5 in steps_values:
            results["has_quick_train"] = True
            results["details"].append("Quick train (5 steps) found")

        # Check for longer train (> 5 steps)
        longer_steps = [s for s in steps_values if s > 5]
        if longer_steps:
            results["has_longer_train"] = True
            results["max_steps_value"] = max(longer_steps)
            results["details"].append(f"Longer train (max_steps={results['max_steps_value']}) found")

    # Check for plotting/monitoring
    if "log_df" in all_source or "plot()" in all_source:
        results["has_plot"] = True
        results["details"].append("Training plot/monitoring found")

    results["passed"] = (
        results["has_grpo_config"] and
        results["has_learning_rate"] and
        results["has_longer_train"] and
        results["max_steps_value"] is not None and
        results["max_steps_value"] >= 50
    )

    return results


def check_final_comparison(notebook: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check Final Comparison.

    Requirements:
    - compare_old_and_new_model function defined
    - Called on at least one dataset example
    - Tests for catastrophic forgetting (general knowledge question)
    """
    results = {
        "passed": False,
        "has_comparison_function": False,
        "has_dataset_test": False,
        "has_knowledge_test": False,
        "has_lora_save": False,
        "details": []
    }

    all_source = get_all_code_source(notebook)

    # Check for comparison function definition
    if "def compare_old_and_new_model" in all_source:
        results["has_comparison_function"] = True
        results["details"].append("compare_old_and_new_model function defined")

    # Check for dataset test (ds[0])
    if re.search(r"ds\[\d+\].*prompt|compare.*ds\[", all_source, re.DOTALL):
        results["has_dataset_test"] = True
        results["details"].append("Dataset item comparison found")

    # Check for general knowledge test
    knowledge_indicators = ["capital", "knowledge", "forgetting", "general"]
    knowledge_source = all_source.lower()
    if any(ind in knowledge_source for ind in knowledge_indicators):
        if "compare_old_and_new_model" in all_source:
            results["has_knowledge_test"] = True
            results["details"].append("General knowledge test found")

    # Check for LoRA save
    if "save_lora" in all_source or "save_pretrained" in all_source:
        results["has_lora_save"] = True
        results["details"].append("LoRA adapter save found")

    results["passed"] = (
        results["has_comparison_function"] and
        results["has_dataset_test"] and
        results["has_knowledge_test"]
    )

    return results


def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*60}{Colors.ENDC}")


def print_check_result(name: str, passed: bool, details: List[str]):
    """Print the result of a rubric check."""
    status = f"{Colors.GREEN}[PASS]{Colors.ENDC}" if passed else f"{Colors.RED}[FAIL]{Colors.ENDC}"
    print(f"\n{status} {Colors.BOLD}{name}{Colors.ENDC}")
    for detail in details:
        bullet = "  " if passed else f"  {Colors.YELLOW}"
        end = "" if passed else Colors.ENDC
        print(f"{bullet}- {detail}{end}")


def print_summary(results: Dict[str, Dict[str, Any]]):
    """Print the overall summary."""
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)

    print_section_header("RUBRIC VALIDATION SUMMARY")

    # Print individual results table
    print(f"\n{'Criterion':<30} {'Status':<10}")
    print("-" * 40)
    for name, result in results.items():
        display_name = name.replace("_", " ").title()
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if result["passed"] else f"{Colors.RED}FAIL{Colors.ENDC}"
        print(f"{display_name:<30} {status}")

    print("-" * 40)

    # Overall result
    if passed_count == total_count:
        print(f"\n{Colors.GREEN}{Colors.BOLD}ALL {total_count} RUBRIC CRITERIA PASSED!{Colors.ENDC}")
        print(f"{Colors.GREEN}Your submission meets all requirements.{Colors.ENDC}")
    else:
        failed = [k.replace("_", " ").title() for k, v in results.items() if not v["passed"]]
        print(f"\n{Colors.RED}{Colors.BOLD}PASSED: {passed_count}/{total_count}{Colors.ENDC}")
        print(f"{Colors.YELLOW}Failed criteria: {', '.join(failed)}{Colors.ENDC}")


def main():
    """Main function to run all rubric checks."""
    # Default notebook path
    notebook_path = "starter/gen_ai_fundamentals_project_starter.ipynb"

    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]

    # Check if notebook exists
    if not Path(notebook_path).exists():
        print(f"{Colors.RED}Error: Notebook not found at {notebook_path}{Colors.ENDC}")
        sys.exit(1)

    print_section_header("TEACHING AN LLM TO REASON")
    print(f"{Colors.CYAN}Rubric Validation Report{Colors.ENDC}")
    print(f"Notebook: {notebook_path}")

    # Load notebook
    notebook = load_notebook(notebook_path)
    print(f"Loaded notebook with {len(notebook.get('cells', []))} cells")

    # Run all checks
    results = {
        "model_setup": check_model_setup(notebook),
        "baseline_prompting": check_baseline_prompting(notebook),
        "reward_design": check_reward_design(notebook),
        "training_monitoring": check_training_monitoring(notebook),
        "final_comparison": check_final_comparison(notebook),
    }

    # Print detailed results
    print_section_header("DETAILED RESULTS")

    check_names = {
        "model_setup": "1. Model Setup (LoRA Configuration)",
        "baseline_prompting": "2. Baseline Prompting (CoT + Example)",
        "reward_design": "3. Reward Design & Validation",
        "training_monitoring": "4. Training & Monitoring",
        "final_comparison": "5. Final Comparison",
    }

    for key, result in results.items():
        print_check_result(check_names[key], result["passed"], result["details"])

    # Print summary
    print_summary(results)

    # Exit with appropriate code
    all_passed = all(r["passed"] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
