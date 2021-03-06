# POS Tagging using Hidden Markov Models
This program implements POS Tagging using Hidden Markov Models and was created in the context of the `CS5012` module. The two algorithms implemented are the Viterbi algorithm and the Beam Search Algorithm. More information about the implementation of this project can be found in the [report](Report.md).

## Usage
Command Line Arguments: python hmm.py <alg_ID> <training_size> <testing_size> <beam_width>(optional)

alg_ID: 1 for Viterbi algorithm, 2 for Beam Search algorithm.

training_size: Number of sentances to be considered for training the algorithm.

testing_size: Number of sentances to be considered when testing the algortithm.

beam_width: Beam width. This parameter should be only be used when using the Beam algorithm.

## Example
```bash
python hmm.py 1 10000 500 
```

This command will run the Viterbi algorithm with 10000 sentances used for training and 500 used for testing.

```bash
python hmm.py 2 20000 5000 4 
```

This command will run the Beam Search algorithm with a beam width of 4. 20000 sentances used for training and 5000 used for testing. 

## References
The [Natural Language toolkit 3.4](https://www.nltk.org/) was used for the creation of this algorithm. The Brown corpus was used for training and testing purposes.
