**AB Testing** is ideal to check if a newly added feature made a significant change.

In the dataset story, a website randomly divides its traffic into two equal parts,
Facebook's current campaign is being served to group A,
Facebook's new campaign is being serverd to group B and we'll check if switching to the new campaign is worth to invest

- The dataset was introduced as csv files, the mean values of the gained price shows no big difference so it's time for 
AB testing. 
- New features have been derived to gain a better insight
- two groups are being concatenated and group names are present in a new variable
- pairplot graphic have been plotted to visually check the differences, there are overlaps
- Shapiro-Wilks Test for Normality have been issued to each group
- Levene Test for variance homogeneity have been issued to the concatenated df.
- the two tests are valid, so we'll continue with Independent Samples T Test 
