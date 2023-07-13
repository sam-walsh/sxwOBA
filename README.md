# [sxwOBA](https://sxwoba.herokuapp.com/) - spray-angle enhanced xwOBA
---

#### What is sxwOBA?
At its core sxwOBA is simply a variation of the popular metric xwOBA. Using three inputs--launch angle, exit velocity, and spray angle.

#### What is wOBA?
---

wOBA (weighted on-base average) is a baseball statistic created by [Tom Tango](http://www.tangotiger.com/index.php) that uses linear weights to measure a player's overall offensive contributions in the form of a rate stat. It is calculated by assigning a weight to each type of offensive event (e.g. single, double, home run, etc.) proportional to their average run values, and then summing the weights for all of a player's offensive events. wOBA can be calculated for any given player's 2022 season by using the following formula (uBB stands for unintentional walks):

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


---
#### What is xwOBA?
---

xwOBA or expected weighted on-base average takes a different approach from wOBA. Rather than crediting hitting outcomes with their linear weights, it credits hitters based on contact quality (specifically exit velocity, launch angle, and&mdash;on some ground balls&mdash;sprint speed). 

Evaluating a hitter's skill based on the quality of their contact with the ball provides a more accurate representation of their abilities because it takes into account the factors that the hitter can control.. On the other hand, the focus on outcomes in wOBA leaves room for variance caused by defense, ballpark dimensions, weather conditions, and other factors that may be attributed to luck. 

For example let's compare two doubles. One is a ground ball down the foul line that is just out of reach of the defender and rolls past the baseinto foul territory, the other is a scorched line drive that shoots past the outfielder's head and bounces on the warning track. wOBA credits each of these doubles equally, because it only recognizes outcomes. xwOBA  ignores the outcome and instead credits each batted ball based on the results of previous, similar batted balls.

More info:
- [Baseball-Savant xwOBA Leaderboards](https://baseballsavant.mlb.com/leaderboard/expected_statistics)
- [xwOBA Definition - MLB.com glossary](https://www.mlb.com/glossary/statcast/expected-woba)



---
#### Where does sxwOBA come in?
---

sxwOBA is a simple variation of xwOBA with the incorporation of spray angle. Spray angle, not to be confused with launch angle, simply corresponds to the horizontal direction of a batted ball. In other words spray angle, which is measured in degrees, tells us the degree to which the batted ball is pulled or to the opposite field.

> *The transformed variable **theta_deg** in my python code measures this angle from home plate with 0 degrees corresponding to the right-field foul line and 90 degrees corresponding to the left-field foul line.*

With its inclusion in the model, we are making the assumption that, like exit velocity and launch angle, hitters also have some degree of control over their spray angle.




