import json, os

if __name__ == '__main__':
    """Load solutions, previously stored as only a vector with process parameters values, and converts them to
    active-inactive format with gene length 1.
    """

    # Paths to experiments with previous solution format to be converted.
    paths = [
        '../../flexbench-data/2_baseline_NSGA-II/2-1_varAnd/logs/',
        '../../flexbench-data/2_baseline_NSGA-II/2-2_tournament/logs/',
        '../../flexbench-data/2_baseline_NSGA-II/2-3_study_eta_varAnd/logs/',
        '../../flexbench-data/2_baseline_NSGA-II/2-4_study_eta_tournament/logs/',
        '../../flexbench-data/2_baseline_NSGA-II/2-5_diverse_init/logs/',
        '../../flexbench-data/2_baseline_NSGA-II/2-6_study_pop_size/logs/',
    ]

    # For each path.
    for path in paths:
        # For each directory inside path.
        directories = os.listdir(path)
        directories.remove('exp_info.txt')
        for directory in directories:
            path_solutions = path + directory + '/'
            path_solutions = path_solutions + os.listdir(path_solutions)[0] + '/solutions/'

            for solutions_i in os.listdir(path_solutions):
                path_solutions_i = path_solutions + solutions_i
                print('Converting %s...' % path_solutions_i)

                # Load from file.
                with open(path_solutions_i, 'r') as f:
                    solutions = json.load(f)

                # Convert to active-inactive format with gene length of 1.
                solutions = [
                    {
                        'solution': sum([[1, param] for param in x['solution']], []),
                        'decoded': x['solution'],
                        'output': x['output'],
                        'feasibility': x['feasibility'],
                        'performance': x['performance'],
                    }
                    for x in solutions
                ]

                # if(directory == 'inconel_718' and solutions_i == 'solutions_repetition_1.json'):
                # print(solutions)

                # Replace original.
                with open(path_solutions_i, 'w') as f:
                    json.dump(solutions, f)
