# Moral Emergence and Change 
These scripts were written to perform 
various analyses on moral sentiment change 
over time as inferred from text. All analyses
can be run from the command-line.

## Usage
Type `--help` in order to display the usage guide for each script. The
purpose of the different analysis files are as follows:
* `analysis/correlate_predictions.py` Correlates the Centroid model's predictions against social/human judgment data.
Use `datatype` option `pew` to replicate the analysis on the Global Trends Pew
research data. We predict a moral sentiment score at both the moral relevance tier and moral polarity tier.
Use `datatype` option `valence` to replicate the correlative analysis between human valence arousal ratings and 
