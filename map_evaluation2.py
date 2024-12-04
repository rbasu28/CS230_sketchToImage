
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


def dict_to_latex_table(data):
    """
    Converts a dictionary to a LaTeX table with rows as iterations and columns as categories.
    The numbers are formatted to 2 decimal places.

    Parameters:
    - data (dict): A nested dictionary where the outer keys represent iterations and
                   the inner keys represent categories.

    Returns:
    - str: A LaTeX table as a string.
    """
    # Extract iteration keys and categories
    iterations = list(data.keys())  # Iterations are rows
    categories = list(next(iter(data.values())).keys())  # Categories are columns

    # Start building the LaTeX table
    latex_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{" + "c|" + "c" * len(iterations) + "}\n"
    latex_table += "\\hline\n"

    # Add header row
    latex_table += "Category & " + " & ".join(str(iteration) for iteration in iterations) + " \\\\\n"
    latex_table += "\\hline\n"

    # Add data rows for each category
    for category in categories:
        row = f"{category} & " + " & ".join(f"{data[iteration][category]:.2f}" for iteration in iterations) + " \\\\\n"
        latex_table += row

    # End the table
    latex_table += "\\hline\n\\end{tabular}\n\\caption{mAP Scores by Iteration and Category}\n\\label{tab:map_scores}\n\\end{table}\n"

    return latex_table


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
Class: pizza, mAP: 0.446258
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
    3: """Class: door, mAP: 0.152605
Class: pizza, mAP: 0.154104
Class: rabbit, mAP: 0.347166
Class: tank, mAP: 0.315701
Class: window, mAP: 0.166938
Class: zebra, mAP: 0.538800
Average test mAP:  0.28311832391446096""",
    # 6 is copied
    4: """Class: door, mAP: 0.234054
Class: pizza, mAP: 0.205863
Class: rabbit, mAP: 0.174764
Class: tank, mAP: 0.341776
Class: window, mAP: 0.205212
Class: zebra, mAP: 0.385301
Average test mAP:  0.2574157207303275""",
    # 8 is copied
    5: """Class: door, mAP: 0.236327
Class: pizza, mAP: 0.190774
Class: rabbit, mAP: 0.296362
Class: tank, mAP: 0.427795
Class: window, mAP: 0.206600
Class: zebra, mAP: 0.584180
Average test mAP:  0.32570658705601635""",
    # 6 is copied
    6 : """Class: door, mAP: 0.316125
Class: pizza, mAP: 0.276623
Class: rabbit, mAP: 0.576310
Class: tank, mAP: 0.520936
Class: window, mAP: 0.449817
Class: zebra, mAP: 0.697173
Average test mAP:  0.4762693950068294""",
    # 10 is copied
    7 : """Class: door, mAP: 0.360449
Class: pizza, mAP: 0.631467
Class: rabbit, mAP: 0.706537
Class: tank, mAP: 0.624823
Class: window, mAP: 0.636815
Class: zebra, mAP: 0.782292
Average test mAP:  0.6270847882938277""",
    # 10 is copied
    8 : """Class: door, mAP: 0.361240
Class: pizza, mAP: 0.679201
Class: rabbit, mAP: 0.879469
Class: tank, mAP: 0.818546
Class: window, mAP: 0.476565
Class: zebra, mAP: 0.771697
Average test mAP:  0.6732388873668973""",
    9 : """Class: door, mAP: 0.369001
Class: pizza, mAP: 0.717706
Class: rabbit, mAP: 0.862363
Class: tank, mAP: 0.897860
Class: window, mAP: 0.568476
Class: zebra, mAP: 0.797403
Average test mAP:  0.7094284488319321""",
    10 : """Class: door, mAP: 0.404212
Class: pizza, mAP: 0.759955
Class: rabbit, mAP: 0.742619
Class: tank, mAP: 0.927720
Class: window, mAP: 0.612041
Class: zebra, mAP: 0.872762
Average test mAP:  0.7243495826380799""",
#     # Rupam's model
#     12 : """Class: door, mAP: 0.370089
# Class: pizza, mAP: 0.720560
# Class: rabbit, mAP: 0.508357
# Class: tank, mAP: 0.528150
# Class: window, mAP: 0.581509
# Class: zebra, mAP: 0.423639
# Average test mAP:  0.5220832288193251""",
}


if __name__ == "__main__":
    map_dict = {key: extract_map_dict(text) for key, text in map_result.items() }

    latex_table_text = dict_to_latex_table(map_dict)
    print(latex_table_text)

    plot_map_lines(map_dict)

