
import matplotlib.pyplot as plt


def extract_map_dict(text):
    """
    Extracts a dictionary from a text containing class names, mAP values, and average mAP.

    Parameters:
    - text (str): The input text containing class names, mAP values, and average mAP.

    Returns:
    - dict: A dictionary with class names as keys and their mAP values as floats,
            and 'average' for the average mAP value.
    """
    result_dict = {}

    for line in text.splitlines():
        # Check if the line contains "Class:" and "mAP:"
        if "Class:" in line and "mAP:" in line:
            # Extract class name and mAP value
            parts = line.split(",")
            class_name = parts[0].split(":")[1].strip()
            map_value = float(parts[1].split(":")[1].strip())
            result_dict[class_name] = map_value
        # Check if the line contains "Average test mAP:"
        elif "Average test mAP:" in line:
            average_value = float(line.split(":")[-1].strip())
            result_dict["[average]"] = average_value

    return result_dict


def plot_map_lines(data_dict):
    """
    Plot multiple lines from a dictionary where keys are iterations and values are sub-dictionaries.
    Includes a point plot for the 'average' value.

    Parameters:
    - data_dict (dict): A dictionary where the outer keys represent iterations,
                        and the inner dictionaries contain key-value pairs to plot.
                        Example: {1: {'door': 0.4, 'pizza': 0.8}, 2: {'door': 0.5, 'pizza': 0.85}}
    """
    # Extract all unique keys from the inner dictionaries
    keys = list(data_dict[next(iter(data_dict))].keys())

    # Prepare data for plotting
    iterations = sorted(list(data_dict.keys()))  # x-axis: num_iterations

    # Plot lines for each key except 'average'
    for key in keys:
        y_values = [data_dict[iteration][key] for iteration in iterations]
        # plt.scatter(iterations, y_values, label=key, zorder=3)
        if key == "[average]":
            plt.plot(iterations, y_values, label=key, marker="o", linewidth=3)
        else:
            plt.plot(iterations, y_values, label=key, marker="o", linestyle='dashed')

    # Remove duplicate labels from the legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # Remove duplicates
    plt.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Plot formatting
    plt.xlabel("Num Iterations")
    plt.ylabel("mAP")
    plt.title("mAP for Different Classes Across Iterations")
    plt.grid(True)
    plt.show()


# epoch: test result
map_result = {
    0: """Class: door, mAP: 0
Class: pizza, mAP: 0
Class: rabbit, mAP: 0
Class: tank, mAP: 0
Class: window, mAP: 0
Class: zebra, mAP: 0
Average test mAP:  0""",
    # 1 is copied
    1: """Class: door, mAP: 0.198021
Class: pizza, mAP: 0.246258
Class: rabbit, mAP: 0.242099
Class: tank, mAP: 0.173594
Class: window, mAP: 0.156064
Class: zebra, mAP: 0.144348
Average test mAP:  0.22818542772346162""",
    # 2 is copied
    2: """Class: door, mAP: 0.176667
Class: pizza, mAP: 0.238478
Class: rabbit, mAP: 0.274916
Class: tank, mAP: 0.111236
Class: window, mAP: 0.189170
Class: zebra, mAP: 0.197860
Average test mAP:  0.1996638983288094""",
    # 4 is copied
    4: """Class: door, mAP: 0.345782
Class: pizza, mAP: 0.310041
Class: rabbit, mAP: 0.480807
Class: tank, mAP: 0.466971
Class: window, mAP: 0.525637
Class: zebra, mAP: 0.225324
Average test mAP:  0.3922954872562048""",
    # 6 is copied
    6: """Class: door, mAP: 0.413362
Class: pizza, mAP: 0.837688
Class: rabbit, mAP: 0.747513
Class: tank, mAP: 0.356898
Class: window, mAP: 0.431146
Class: zebra, mAP: 0.284562
Average test mAP:  0.5176935002037517""",
    # 8 is copied
    8: """Class: door, mAP: 0.360232
Class: pizza, mAP: 0.635535
Class: rabbit, mAP: 0.756800
Class: tank, mAP: 0.701794
Class: window, mAP: 0.446198
Class: zebra, mAP: 0.362590
Average test mAP:  0.5502283593759425""",
    # 10 is copied
    10 : """Class: door, mAP: 0.402829
Class: pizza, mAP: 0.635646
Class: rabbit, mAP: 0.619019
Class: tank, mAP: 0.514493
Class: window, mAP: 0.552948
Class: zebra, mAP: 0.586539
Average test mAP:  0.5542191604244185""",
    # Rupam's model
    12 : """Class: door, mAP: 0.370089
Class: pizza, mAP: 0.720560
Class: rabbit, mAP: 0.508357
Class: tank, mAP: 0.528150
Class: window, mAP: 0.581509
Class: zebra, mAP: 0.423639
Average test mAP:  0.5220832288193251""",
#     # Rupam's model
#     50 : """Class: door, mAP: 0.508660
# Class: pizza, mAP: 0.958209
# Class: rabbit, mAP: 0.942717
# Class: tank, mAP: 0.966872
# Class: window, mAP: 0.463169
# Class: zebra, mAP: 0.922981
# Average test mAP:  0.8033444368506977""",
}


if __name__ == "__main__":
    map_dict = {key: extract_map_dict(text) for key, text in map_result.items() }
    print(map_dict)
    plot_map_lines(map_dict)
