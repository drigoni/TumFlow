import sys

from mflow.optimize_cancer_property import MoFlowProp

if __name__ == "__main__":
    print("parameters:", sys.argv)
    # Check if the script is called with the correct number of arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        # remove first parameter, so that the functions will access the right values
        sys.argv.pop(0)
        if mode == "preprocess":
            from data import data_preprocess
            data_preprocess.preprocess()
        elif mode == "train":
            from mflow import train_model
            train_model.train()
        elif mode == "train_optimizer":
            from mflow import optimize_cancer_property
            optimize_cancer_property.main()
        elif mode == "generate":
            from mflow import generate
            generate.main()
        elif mode == "optimize":
            from mflow import optimize_cancer_property
            optimize_cancer_property.main()
        else:
            print("Invalid mode {}. Only choices [preprocess, train, train_optimizer, generate, optimize]".format(mode))
    else:
        print("Invalid number of arguments. Select one of the mode in: [preprocess, train, train_optimizer, generate, optimize]. ")
