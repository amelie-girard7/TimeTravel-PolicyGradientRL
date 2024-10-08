import json
import math

def calculate_statistics(file_path):
    lengths = []
    shortest_ending = None
    longest_ending = None

    with open(file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line)
                original_ending = data.get('edited_ending', '')
                story_id = data.get('story_id', 'Unknown')

                length = len(original_ending)
                lengths.append(length)

                if shortest_ending is None or length < len(shortest_ending['text']):
                    shortest_ending = {'story_id': story_id, 'text': original_ending, 'length': length}

                if longest_ending is None or length > len(longest_ending['text']):
                    longest_ending = {'story_id': story_id, 'text': original_ending, 'length': length}

            except json.JSONDecodeError as e:
                continue

    if not lengths:
        return 0, 0, None, None

    average_length = sum(lengths) / len(lengths)
    variance = sum((x - average_length) ** 2 for x in lengths) / len(lengths)
    std_deviation = math.sqrt(variance)

    return average_length, std_deviation, shortest_ending, longest_ending

# File path to the JSON file
file_path = '/data/agirard/Projects/Timetravel/data/transformed/test_data_sample.json'

# Calculate and print the statistics
average_length, std_deviation, shortest_ending, longest_ending = calculate_statistics(file_path)

print(f'Average number of characters in edited_ending: {average_length:.2f}')
print(f'Standard deviation of characters in edited_ending: {std_deviation:.2f}')
if shortest_ending:
    print(f'Shortest edited_ending (Story ID: {shortest_ending["story_id"]}): {shortest_ending["text"]} with {shortest_ending["length"]} characters')
if longest_ending:
    print(f'Longest edited_ending (Story ID: {longest_ending["story_id"]}): {longest_ending["text"]} with {longest_ending["length"]} characters')
