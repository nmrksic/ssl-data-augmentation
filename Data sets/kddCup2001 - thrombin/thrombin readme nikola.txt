This data was downloaded from:  http://pages.cs.wisc.edu/~dpage/kddcup2001/

Task 1 - binding to Thrombin

The data is ~500MB when extracted - Matlab can't handle it in that form. 

I initially processed data using C++, creating a list of features present in the data that was then used in Matlab to reconstruct the sparse data. Data was reordered and compiled into thrombin.mat, which can easily be imported into Matlab.


Furthermore, the test set, which contained the features of the compounds to be classified as active/inactive, was also quite big ~ 170MB, processed by C++ in the same way. 



Regarding Dataset 1, can we expect that the ratio of the number of active compounds to the number of inactive compunds in the final test set to be roughly same as the ratio in the given training set?

No, and to keep this similar to the real-world scenario, we don't want to say what the ratio is. But being generous people :-), we will give away 2 items of information. (1) The compounds in the test set were made after chemists saw the activity results for the training set, so as you might expect the test set has a higher fraction of actives than did the training set. (2) Nevertheless, the test set still has more inactives than actives. We realize our method of testing makes the task tougher than it would be with the standard assumption that test data are drawn according to the same distribution as the training data. But this is a common scenario for those working in the pharmaceutical industry.


For Dataset 1, assume that the test set contains NA active substances and NI inactive ones. My procedure classifies correctly NAcor of NA active substances and NIcor of NI inactive substances. Is it correct that the measure of error of my procedure would be Err = (NA - NAcorr)/NA + (NI - NIcor)/NI ? Is it Err I should minimize?

Yes. The person/group that minimizes this on the test set wins. This is equivalent to minimizing the ordinary error with differential costs. For example, if the test set contains 10 actives and 100 inactives, we want to maximize accuracy (minimize ordinary error) where each active counts 10 times but each inactive only once. This is the standard accuracy with differential misclassification costs as is used throughout data mining and machine learning.



I was looking at the Thrombin data set, and found that 593 of the 1909 samples are all zero (i.e., none of the bits are 1). Included in this set of 593 are two Active compounds. Is this correct, or could it be a mistake?

This is correct. Clearly one cannot get 100% training set accuracy because of this, but one probably can remove the two all-zero actives from the training set without incurring any disadvantage. I would not suggest removing any of the all-zero inactives, since their contributions to the frequencies of the different attributes are important (at least for frequency-based approaches such as decision/classification trees).









