# Text inference of moral sentiment change
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
Use `datatype` option `valence` to replicate the correlative analysis between human valence arousal ratings and moral 
predictions. Produces a `csv` file.

* `analysis/plot_time_courses.py` Create time course diagrams at each tier in the moral sentiment framework

* `analysis/top_k_retrievals.py` Fetch the top changing words (measured according to the slope of their moral 
trajectories) at the polarity and relevance tiers.

