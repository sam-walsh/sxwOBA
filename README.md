
sxwOBA - Spray-Angle Enhanced xwOBA

This project is focused on the development of a variation of the popular baseball metric xwOBA, called sxwOBA. The sxwOBA metric incorporates four inputs: launch angle, exit velocity, spray angle, and sprint speed to provide a more comprehensive measure of a player's offensive contributions.

What is wOBA?

Weighted On-Base Average (wOBA) is a baseball statistic that measures a player's overall offensive contributions. It assigns a weight to each type of offensive event (e.g., single, double, home run, etc.) proportional to their average run values, and then sums the weights for all of a player's offensive events.

$$ \text{wOBA} = \frac{0.69 \cdot \text{uBB} + 0.72 \cdot \text{HBP} + 0.88 \cdot \text{1B} + 1.26 \cdot \text{2B} + 1.60 \cdot \text{3B} + 2.07 \cdot \text{HR} }{\text{AB + BB + SF + HBP - IBB }} $$

wOBA is a more accurate representation of a hitter's contribution to scoring than other metrics such as On-base percentage (**OBP**) and Slugging percentage (**SLG**).
- OBP does not differentiate between different ways of getting on base. A walk is treated equally to a home run.
- SLG inaccurately treats home runs as having four times the value of a single. It also ignores walks altogether.
- On-base plus slugging (OPS) attempts to alleviate the deficiencies of each of these metrics by simply adding them together into one metric, however this is a quick and dirty estimate of run production that improves upon, but **does not** fix the deficiencies of each of the two metrics. It is still inaccurately weighting the value of each hit type, and overvalues the role of SLG in run production.

> *wOBA is conveniently scaled the same as on-base percentage. Therefore, **league-average wOBA is equivalent to league-average OBP** allowing the metric to be easily understood by those familar to OBP.*

More info:
- [Season-by-season linear weights](https://www.fangraphs.com/guts.aspx?type=cn)
- [Calculating linear weights yourself](http://www.insidethebook.com/ee/index.php/site/article/woba_year_by_year_calculations/)
- [wOBA - Fangraphs glossary](https://library.fangraphs.com/offense/woba/)

What is xwOBA?

Expected Weighted On-Base Average (xwOBA) takes a different approach from wOBA. It credits hitters based on contact quality (specifically exit velocity, launch angle, and, on some ground balls, sprint speed). This provides a more accurate representation of a hitter's abilities as it takes into account the factors that the hitter can control.
What is sxwOBA?

More info:
- [Baseball-Savant xwOBA Leaderboards](https://baseballsavant.mlb.com/leaderboard/expected_statistics)
- [xwOBA Definition - MLB.com glossary](https://www.mlb.com/glossary/statcast/expected-woba)


sxwOBA is a simple variation of xwOBA with the incorporation of spray angle. Spray angle corresponds to the horizontal direction of a batted ball. With its inclusion in the model, we are making the assumption that, like exit velocity and launch angle, hitters also have some degree of control over their spray angle.
Project Structure

The project contains several Jupyter notebooks and Python scripts used for data analysis, modeling, and data manipulation:

- models.ipynb: This notebook contains code for importing necessary libraries, loading data, and setting up models.
- sandbox.ipynb: This notebook is used for exploratory data analysis and testing code snippets.
- test.ipynb: This notebook is used for testing purposes.
- re24.ipynb: This notebook contains code for data manipulation and analysis.
- modeling.ipynb: This notebook contains code for data modeling.
- get_leaders.py: This Python script is used to fetch and process data.
- leaders.ipynb: This notebook is used to analyze and visualize the leaders in various metrics.
Getting Started

To get started with this project, you need to have Python installed along with several libraries including pandas, numpy, matplotlib, seaborn, pybaseball, sklearn, and xgboost. You can clone the repository and run the notebooks using Jupyter Notebook.

Authors

- Sam Walsh

License

This project is licensed under the MIT License - see the LICENSE.md file for details.
