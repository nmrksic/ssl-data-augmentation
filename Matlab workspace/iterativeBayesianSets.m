function entitySet = iterativeBayesianSets(FeatureMatrix, QuerySet, NegativeSet)

% to return the set of new positives!

% QuerySet contains the (known) positive set.
% iterativeBayes implements blind relevance feedback to Bayesian sets.

% NumEntries is the total number of positives and negatives, i.e. size of
% redux set, which we need to estimate number of positives to extract.

Labels = zeros(size(FeatureMatrix, 1), 1);

Labels(QuerySet) = 1;


%TODO: This is total number - we need to distinguish w.r.t. size of
%unlabeled set!!! TODO TODO TODO
%N = round((size(FeatureMatrix, 1) * length(QuerySet) / NumEntries));

topK = max(round(length(QuerySet)/20), 5); % this is the topK we extract in different iterations

% we want to label new ones repeatedly, until running Bayesian sets with
% our labels produces correct topK!

while(true)
    
    disp('tag')
    
    scores = BayesianSet(FeatureMatrix, LabelToArray(Labels), 2);
   
    [~, idx] = sort(scores, 'descend');
    
    sumOld = sum(Labels);
    
    Labels( idx( 1:min(topK, length(idx)) )) = 1; % label the new ones as positive, to use for new scoring!
    %  Labels(NegativeSet) = 0; % issue #1 related - will this prevent convergence??
    if( sum(Labels) == sumOld) 
        break; 
    end % system converged, no new positives in top N, return

end

entitySet = setdiff( LabelToArray(Labels), QuerySet );
entitySet = setdiff(entitySet, NegativeSet); % should we be modifying these - issue #1

NumOfNewElements = length(entitySet)

% it would make very much sense to exclude those known to be negative!!! TODO TODO TODO TODO TODO 