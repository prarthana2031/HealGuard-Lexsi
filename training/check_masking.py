"""Run the mask check on the training data. 30 seconds, before any GPU spend.

    python -m training.check_masking
"""

from training.masking import main

if __name__ == "__main__":
    main()
