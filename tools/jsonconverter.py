import json

def get_student_info(json_file, search_param):
    try:
        # Load the JSON data from a file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON data: {e}")
        return None

    # Search for the student information
    for student in data:
        try:
            if student["student_code"] == search_param:
                return student["name"], student["id"]
        except KeyError:
            print("Invalid JSON structure")
            return None

    # Return None if no matching student is found
    return None