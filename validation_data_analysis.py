import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm

# Path definitions
REASONING_VALIDATION_PATH = 'data/reasoning/validation_interactive'
OUTPUT_DIR = 'outputs/validation_analysis'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_reasoning_validation_data(max_files=None):
    """Load all reasoning validation data from JSON files."""
    print(f"Loading reasoning validation data from {REASONING_VALIDATION_PATH}...")
    
    # Get all JSON files
    json_files = [os.path.join(REASONING_VALIDATION_PATH, f) for f in os.listdir(REASONING_VALIDATION_PATH) 
                 if f.endswith('.json')]
    
    if max_files:
        json_files = json_files[:max_files]
    
    print(f"Found {len(json_files)} JSON files.")
    
    # Load data
    validation_data = []
    
    for file_path in tqdm(json_files, desc="Loading files"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Add source file to track which file this came from
            data['source_file'] = os.path.basename(file_path)
            validation_data.append(data)
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(validation_data)} validation scenarios.")
    return validation_data

def extract_basic_stats(data):
    """Extract basic statistics from the validation data."""
    stats = {
        'total_scenarios': len(data),
        'unique_scenario_ids': len(set([d['sid'] for d in data])),
        'avg_env_questions': np.mean([len(d.get('env_q', [])) for d in data]),
        'avg_ego_questions': np.mean([len(d.get('ego_q', [])) for d in data]),
        'avg_sur_questions': np.mean([len(d.get('sur_q', [])) for d in data]),
        'avg_int_questions': np.mean([len(d.get('int_q', [])) for d in data]),
        'total_questions': sum([len(d.get('env_q', [])) + len(d.get('ego_q', [])) + 
                              len(d.get('sur_q', [])) + len(d.get('int_q', [])) for d in data]),
    }
    
    # Count files with/without surrounding agents
    stats['scenarios_with_surrounding_agents'] = sum([1 for d in data if len(d.get('sur_q', [])) > 0])
    stats['scenarios_without_surrounding_agents'] = stats['total_scenarios'] - stats['scenarios_with_surrounding_agents']
    
    # Calculate percentages
    stats['percent_with_surrounding'] = stats['scenarios_with_surrounding_agents'] / stats['total_scenarios'] * 100
    stats['percent_without_surrounding'] = stats['scenarios_without_surrounding_agents'] / stats['total_scenarios'] * 100
    
    return stats

def analyze_question_types(data):
    """Analyze the types of questions asked in the validation set."""
    env_questions = []
    ego_questions = []
    sur_questions = []
    int_questions = []
    
    for scenario in data:
        env_questions.extend(scenario.get('env_q', []))
        ego_questions.extend(scenario.get('ego_q', []))
        sur_questions.extend(scenario.get('sur_q', []))
        int_questions.extend(scenario.get('int_q', []))
    
    # Get the most common questions for each category
    env_common = Counter(env_questions).most_common(10)
    ego_common = Counter(ego_questions).most_common(10)
    sur_common = Counter(sur_questions).most_common(10)
    int_common = Counter(int_questions).most_common(10)
    
    return {
        'env_common': env_common,
        'ego_common': ego_common,
        'sur_common': sur_common,
        'int_common': int_common,
    }

def analyze_answer_patterns(data):
    """Analyze common patterns in the answers."""
    # Mapping of question categories to their answer fields
    qa_pairs = [
        ('env_q', 'env_a'),
        ('ego_q', 'ego_a'),
        ('sur_q', 'sur_a'),
        ('int_q', 'int_a'),
    ]
    
    # Store question-answer pairs for analysis
    qa_mapping = defaultdict(list)
    
    for scenario in data:
        for q_field, a_field in qa_pairs:
            questions = scenario.get(q_field, [])
            answers = scenario.get(a_field, [])
            
            # Make sure we have the same number of questions and answers
            if len(questions) == len(answers):
                for q, a in zip(questions, answers):
                    qa_mapping[q].append(a)
    
    # Find questions with consistent answers
    consistent_answers = {}
    
    for question, answers in qa_mapping.items():
        if len(answers) > 1:
            answer_counter = Counter(answers)
            most_common_answer, count = answer_counter.most_common(1)[0]
            consistency = count / len(answers)
            
            if consistency > 0.8 and len(answers) >= 5:  # Consider it consistent if same answer appears >80% of time
                consistent_answers[question] = {
                    'common_answer': most_common_answer,
                    'consistency': consistency,
                    'occurrences': len(answers)
                }
    
    # Sort by number of occurrences
    sorted_consistent = dict(sorted(consistent_answers.items(), 
                                  key=lambda x: x[1]['occurrences'], 
                                  reverse=True))
    
    return {
        'qa_mapping': qa_mapping,
        'consistent_answers': sorted_consistent
    }

def analyze_interaction_patterns(data):
    """Analyze patterns in the interaction descriptions."""
    interaction_answers = []
    
    for scenario in data:
        int_answers = scenario.get('int_a', [])
        interaction_answers.extend(int_answers)
    
    # Extract common phrases from interaction answers
    common_interactions = Counter()
    
    # Common driving actions to look for
    driving_actions = ['yield', 'stop', 'continue', 'accelerate', 'decelerate', 
                      'turn left', 'turn right', 'merge', 'change lane', 'brake']
    
    for answer in interaction_answers:
        answer_lower = answer.lower()
        for action in driving_actions:
            if action in answer_lower:
                common_interactions[action] += 1
    
    # Return most common interaction patterns
    interaction_patterns = dict(common_interactions.most_common())
    
    return {
        'interaction_patterns': interaction_patterns
    }

def visualize_statistics(stats, question_types, interaction_patterns):
    """Generate visualizations of the statistics."""
    # 1. Pie chart of scenarios with/without surrounding agents
    plt.figure(figsize=(10, 6))
    plt.pie([stats['scenarios_with_surrounding_agents'], stats['scenarios_without_surrounding_agents']], 
            labels=['With surrounding agents', 'Without surrounding agents'],
            autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Scenarios by Presence of Surrounding Agents')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/scenario_distribution_pie.png')
    plt.close()
    
    # 2. Bar chart of average questions per category
    categories = ['Environment', 'Ego Agent', 'Surrounding Agents', 'Interaction']
    avg_questions = [stats['avg_env_questions'], stats['avg_ego_questions'], 
                    stats['avg_sur_questions'], stats['avg_int_questions']]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, avg_questions, color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Average Number of Questions per Category')
    plt.ylabel('Average Number of Questions')
    plt.ylim(0, max(avg_questions) * 1.2)  # Add some space for the text
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/avg_questions_per_category.png')
    plt.close()
    
    # 3. Bar chart of most common interaction patterns
    if interaction_patterns['interaction_patterns']:
        plt.figure(figsize=(14, 8))
        patterns = list(interaction_patterns['interaction_patterns'].keys())
        counts = list(interaction_patterns['interaction_patterns'].values())
        
        # Sort by count
        sorted_indices = np.argsort(counts)[::-1]
        patterns = [patterns[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        
        # Take top 10 for better visualization
        patterns = patterns[:10]
        counts = counts[:10]
        
        bars = plt.bar(patterns, counts, color='#9b59b6')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height}', ha='center', va='bottom')
        
        plt.title('Most Common Interaction Patterns')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/common_interaction_patterns.png')
        plt.close()

def main(max_files=None):
    """Main function to analyze validation data."""
    # Load data
    validation_data = load_reasoning_validation_data(max_files)
    
    if not validation_data:
        print("No validation data loaded. Exiting.")
        return
    
    # Extract basic statistics
    print("\nExtracting basic statistics...")
    stats = extract_basic_stats(validation_data)
    
    # Print statistics
    print("\nBasic Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Analyze question types
    print("\nAnalyzing question types...")
    question_types = analyze_question_types(validation_data)
    
    # Print most common questions
    print("\nMost common environment questions:")
    for q, count in question_types['env_common']:
        print(f"  {q} ({count} occurrences)")
    
    print("\nMost common ego agent questions:")
    for q, count in question_types['ego_common']:
        print(f"  {q} ({count} occurrences)")
    
    print("\nMost common surrounding agent questions:")
    for q, count in question_types['sur_common']:
        print(f"  {q} ({count} occurrences)")
    
    print("\nMost common interaction questions:")
    for q, count in question_types['int_common']:
        print(f"  {q} ({count} occurrences)")
    
    # Analyze answer patterns
    print("\nAnalyzing answer patterns...")
    answer_patterns = analyze_answer_patterns(validation_data)
    
    # Print questions with consistent answers
    print("\nQuestions with consistent answers:")
    for q, info in list(answer_patterns['consistent_answers'].items())[:10]:  # Show top 10
        print(f"  Q: {q}")
        print(f"     A: {info['common_answer']} " +
              f"(consistency: {info['consistency']:.2f}, occurrences: {info['occurrences']})")
    
    # Analyze interaction patterns
    print("\nAnalyzing interaction patterns...")
    interaction_patterns = analyze_interaction_patterns(validation_data)
    
    # Print common interaction patterns
    print("\nCommon interaction patterns:")
    for action, count in interaction_patterns['interaction_patterns'].items():
        print(f"  {action}: {count} occurrences")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_statistics(stats, question_types, interaction_patterns)
    
    # Save all statistics to JSON
    output = {
        'basic_stats': stats,
        'question_types': {
            'env_common': dict(question_types['env_common']),
            'ego_common': dict(question_types['ego_common']),
            'sur_common': dict(question_types['sur_common']),
            'int_common': dict(question_types['int_common']),
        },
        'consistent_answers': {q: info for q, info in list(answer_patterns['consistent_answers'].items())[:20]},
        'interaction_patterns': interaction_patterns['interaction_patterns'],
    }
    
    with open(f'{OUTPUT_DIR}/validation_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {OUTPUT_DIR}/validation_analysis.json")
    print(f"Visualizations saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main(max_files=None)  # Set to a number to limit processing, or None to process all files 