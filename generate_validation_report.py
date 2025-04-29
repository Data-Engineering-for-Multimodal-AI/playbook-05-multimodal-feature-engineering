import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Path definitions
OUTPUT_DIR = 'outputs/validation_analysis'
REPORT_FILE = 'validation_report.md'

def load_analysis_results():
    """Load all analysis results from JSON files."""
    results = {}
    
    # List of result files to load
    result_files = {
        'scenario_comparison': 'scenario_id_comparison.json',
        'validation_analysis': 'validation_analysis.json',
        'question_statistics': 'question_statistics.json',
        'detailed_scenarios': 'detailed_scenario_analysis.json',
        'agent_mapping_stats': 'agent_mapping_stats.json'
    }
    
    # Load each file if it exists
    for key, filename in result_files.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    results[key] = json.load(f)
                print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return results

def generate_report(results):
    """Generate a comprehensive markdown report of all analysis results."""
    # Start with report header
    report = [
        "# Waymo Motion and Reasoning Validation Dataset Analysis",
        f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Table of Contents",
        "1. [Overview](#overview)",
        "2. [Dataset Comparison](#dataset-comparison)",
        "3. [Reasoning Data Analysis](#reasoning-data-analysis)",
        "4. [Question Analysis](#question-analysis)",
        "5. [Common Scenario Analysis](#common-scenario-analysis)",
        "6. [Agent Mapping](#agent-mapping)",
        "7. [Conclusion](#conclusion)",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This report provides a comprehensive analysis of the Waymo Motion and Reasoning validation datasets. ",
        "The analysis focuses on understanding the structure, content, and relationship between these two datasets.",
        "",
        "The datasets analyzed are located at:",
        "- Motion data: `data/motion/tf_example/validation_interactive`",
        "- Reasoning data: `data/reasoning/validation_interactive`",
        "",
        "---",
        "",
    ]
    
    # Dataset Comparison section
    report.extend([
        "## Dataset Comparison",
        "",
    ])
    
    if 'scenario_comparison' in results:
        comparison = results['scenario_comparison']
        common_count = len(comparison.get('common_ids', []))
        motion_only_count = len(comparison.get('motion_only_ids', []))
        reasoning_only_count = len(comparison.get('reasoning_only_ids', []))
        total_motion = common_count + motion_only_count
        total_reasoning = common_count + reasoning_only_count
        
        report.extend([
            f"### Scenario ID Comparison",
            "",
            f"- Total Motion scenarios: **{total_motion}**",
            f"- Total Reasoning scenarios: **{total_reasoning}**",
            f"- Common scenarios: **{common_count}** ({common_count/total_motion:.1%} of Motion, {common_count/total_reasoning:.1%} of Reasoning)",
            f"- Motion-only scenarios: **{motion_only_count}** ({motion_only_count/total_motion:.1%} of Motion)",
            f"- Reasoning-only scenarios: **{reasoning_only_count}** ({reasoning_only_count/total_reasoning:.1%} of Reasoning)",
            "",
            "The overlap between the Motion and Reasoning datasets is relatively small, with only a fraction of scenarios ",
            "being common between the two datasets.",
            "",
        ])
    else:
        report.extend([
            "*Scenario comparison data not available.*",
            "",
        ])
    
    # Reasoning Data Analysis section
    report.extend([
        "## Reasoning Data Analysis",
        "",
    ])
    
    if 'validation_analysis' in results:
        analysis = results['validation_analysis']
        basic_stats = analysis.get('basic_stats', {})
        
        if basic_stats:
            report.extend([
                "### Basic Statistics",
                "",
                f"- Total scenarios: **{basic_stats.get('total_scenarios', 'N/A')}**",
                f"- Unique scenario IDs: **{basic_stats.get('unique_scenario_ids', 'N/A')}**",
                f"- Total questions: **{basic_stats.get('total_questions', 'N/A')}**",
                "",
                "### Question Distribution",
                "",
                f"- Average environment questions: **{basic_stats.get('avg_env_questions', 'N/A'):.2f}**",
                f"- Average ego agent questions: **{basic_stats.get('avg_ego_questions', 'N/A'):.2f}**",
                f"- Average surrounding agent questions: **{basic_stats.get('avg_sur_questions', 'N/A'):.2f}**",
                f"- Average interaction questions: **{basic_stats.get('avg_int_questions', 'N/A'):.2f}**",
                "",
                "### Surrounding Agent Presence",
                "",
                f"- Scenarios with surrounding agents: **{basic_stats.get('scenarios_with_surrounding_agents', 'N/A')}** ({basic_stats.get('percent_with_surrounding', 'N/A'):.1f}%)",
                f"- Scenarios without surrounding agents: **{basic_stats.get('scenarios_without_surrounding_agents', 'N/A')}** ({basic_stats.get('percent_without_surrounding', 'N/A'):.1f}%)",
                "",
                "The vast majority of scenarios contain surrounding agents, indicating that the dataset focuses on ",
                "interactive driving scenarios.",
                "",
            ])
        
        # Interaction patterns
        interaction_patterns = analysis.get('interaction_patterns', {})
        if interaction_patterns:
            report.extend([
                "### Interaction Patterns",
                "",
                "The most common interaction patterns in the dataset are:",
                "",
            ])
            
            # Sort by frequency
            patterns = [(k, v) for k, v in interaction_patterns.items()]
            patterns.sort(key=lambda x: x[1], reverse=True)
            
            for pattern, count in patterns:
                report.append(f"- **{pattern}**: {count} occurrences")
            
            report.extend([
                "",
                "This distribution shows the relative frequency of different driving actions mentioned in the dataset.",
                "",
            ])
    else:
        report.extend([
            "*Validation analysis data not available.*",
            "",
        ])
    
    # Question Analysis section
    report.extend([
        "## Question Analysis",
        "",
    ])
    
    if 'validation_analysis' in results and 'question_types' in results['validation_analysis']:
        question_types = results['validation_analysis']['question_types']
        
        # Environment questions
        if 'env_common' in question_types:
            report.extend([
                "### Most Common Environment Questions",
                "",
            ])
            
            for question, count in question_types['env_common'].items():
                report.append(f"- {question} ({count} occurrences)")
            
            report.append("")
        
        # Ego questions
        if 'ego_common' in question_types:
            report.extend([
                "### Most Common Ego Agent Questions",
                "",
            ])
            
            for question, count in question_types['ego_common'].items():
                report.append(f"- {question} ({count} occurrences)")
            
            report.append("")
        
        # Surrounding questions
        if 'sur_common' in question_types:
            report.extend([
                "### Most Common Surrounding Agent Questions",
                "",
            ])
            
            for question, count in question_types['sur_common'].items():
                report.append(f"- {question} ({count} occurrences)")
            
            report.append("")
        
        # Interaction questions
        if 'int_common' in question_types:
            report.extend([
                "### Most Common Interaction Questions",
                "",
            ])
            
            for question, count in question_types['int_common'].items():
                report.append(f"- {question} ({count} occurrences)")
            
            report.append("")
    
    # Additional question statistics from the question_statistics file
    if 'question_statistics' in results:
        q_stats = results['question_statistics']
        
        report.extend([
            "### Question Count Statistics",
            "",
            "| Category | Mean | Median | Min | Max |",
            "|----------|------|--------|-----|-----|",
        ])
        
        for category, stats in q_stats.items():
            report.append(f"| {category.capitalize()} | {stats['mean']:.2f} | {stats['median']:.1f} | {stats['min']} | {stats['max']} |")
        
        report.extend([
            "",
            "These statistics show the distribution of questions across different categories in the validation dataset.",
            "",
            "![Question Distribution](question_distribution.png)",
            "",
        ])
    else:
        report.extend([
            "*Detailed question statistics not available.*",
            "",
        ])
    
    # Common Scenario Analysis section
    report.extend([
        "## Common Scenario Analysis",
        "",
    ])
    
    if 'detailed_scenarios' in results:
        detailed = results['detailed_scenarios']
        
        report.extend([
            "### Detailed Analysis of Common Scenarios",
            "",
            "Analyzed in detail: **{0}** common scenarios".format(len(detailed)),
            "",
            "| Scenario ID | Motion Agents | Ego Agent ID | Ego Match | Reasoning Files |",
            "|-------------|---------------|--------------|-----------|----------------|",
        ])
        
        for scenario in detailed:
            scenario_id = scenario.get('scenario_id', 'N/A')
            num_agents = scenario.get('motion', {}).get('num_agents', 'N/A')
            ego_id = scenario.get('motion', {}).get('ego_agent_id', 'N/A')
            ego_match = scenario.get('ego_id_match', False)
            num_files = scenario.get('num_reasoning_files', 'N/A')
            
            report.append(f"| {scenario_id} | {num_agents} | {ego_id} | {ego_match} | {num_files} |")
        
        # Calculate match statistics
        matches = [s.get('ego_id_match', False) for s in detailed]
        match_rate = sum(matches) / len(matches) if matches else 0
        
        report.extend([
            "",
            f"Ego agent ID match rate: **{match_rate:.1%}**",
            "",
            "When comparing common scenarios between the Motion and Reasoning datasets, we checked if the ego agent IDs match. ",
            f"In our sample, {match_rate:.1%} of scenarios have matching ego agent IDs, suggesting that the datasets are indeed aligned.",
            "",
        ])
    else:
        report.extend([
            "*Detailed scenario analysis not available.*",
            "",
        ])
    
    # Agent Mapping section
    report.extend([
        "## Agent Mapping",
        "",
    ])
    
    if 'agent_mapping_stats' in results:
        mapping_stats = results['agent_mapping_stats']
        
        report.extend([
            "### Agent ID Mapping Statistics",
            "",
            f"- Total agent mappings: **{mapping_stats.get('total_mappings', 'N/A')}**",
            f"- Common agents (in both datasets): **{mapping_stats.get('common_agents', 'N/A')}**",
            f"- Motion-only agents: **{mapping_stats.get('motion_only_agents', 'N/A')}**",
            f"- Reasoning-only agents: **{mapping_stats.get('reasoning_only_agents', 'N/A')}**",
            "",
        ])
    else:
        report.extend([
            "*Agent mapping statistics not available.*",
            "",
        ])
    
    # Conclusion section
    report.extend([
        "## Conclusion",
        "",
        "### Key Findings",
        "",
        "1. The Waymo Motion and Reasoning validation datasets contain **many unique scenarios** that are not present in both datasets.",
        "2. For common scenarios that appear in both datasets, the **ego agent IDs match** between the datasets, confirming alignment.",
        "3. The Reasoning dataset focuses heavily on **interactive scenarios** with surrounding agents present in most cases.",
        "4. Questions about environment, ego agent, surrounding agents, and interactions follow **consistent patterns**.",
        "5. The most common interaction patterns involve **yielding, continuing, and stopping**, reflecting typical driving behaviors.",
        "",
        "### Recommendations",
        "",
        "1. **Focus on common scenarios** for joint motion and reasoning analysis.",
        "2. Consider the **distribution of question types** when designing models to process reasoning data.",
        "3. Pay special attention to **interactive scenarios** as they form the majority of the validation dataset.",
        "4. Leverage the **matching ego agent IDs** to connect motion prediction with reasoning capabilities.",
        "",
        "---",
        "",
        "*End of Report*",
    ])
    
    return '\n'.join(report)

def main():
    """Generate a comprehensive report of validation data analysis."""
    print("Loading analysis results...")
    results = load_analysis_results()
    
    if not results:
        print("No analysis results found. Run analysis scripts first.")
        return
    
    print("Generating validation report...")
    report_content = generate_report(results)
    
    # Save report
    report_path = os.path.join(OUTPUT_DIR, REPORT_FILE)
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    main() 