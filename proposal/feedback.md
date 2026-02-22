This is a good proposal, if a bit ambitious.  I think your goal here is probably going to be to predict surge size, and you'd like to do that as early as possible.  So, given an uptick in cases, how does the time of the forecast tradeoff against accuracy (total number of cases).

To do this, you're going to need to identify a large number of potential swells - these will likely be identifiable at the regional level - I imagine there is a type of thresholding that occurs, where-in a contagion accelerates in one region and only "jumps" to other regions when it is particularly contagious.  

The thing that makes this particularly tricky is that different diseases operate in different ways - critically, contagions have different modes of spreading (airborne, contact, ingested, etc.) and different levels of infectiousness.  This intersects with the nature of the network through which it spreads - how mobile is the population, how densely populated is an area.

There are lots of possibilities here, but you might want to have a look at your data and focus on a specific class of disease (e.g., flu, covid, other) in a particular region (Australia and New Zealand are nice because they are relatively isolated).  You might also want to consider explicitly modeling the infectiousness (R_0, the basic reproduction number) and then use that, along with inferences about contact rates (geographically variable) to come up with a model that is then used for prediction.  There's a lot of literature on this - much can be found in basic epidemiology textbooks.

So, I think the problem is a good one, but it's a simple ML problem - you're going to need to learn a lot here!

SCORE: 5/5