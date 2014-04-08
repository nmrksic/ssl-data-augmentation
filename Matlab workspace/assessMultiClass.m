function [augmentedResults, originalResults] = assessMultiClass(TrainFM, TrainLabels, TestFM, TestLabels, UnlabeledFM, kkdThreshold)

numClasses = length(unique(TestLabels));

originalLabeling = MultiClassSVM(TrainFM, TrainLabels, TestFM, 0);

originalResults = quality(originalLabeling, TestLabels);

% We need to extract new positives of every class!

newPositives = cell(numClasses);

for i = 1 : numClasses 
    
    idxAccess = LabelToArray((TrainLabels == i));
    
    augFM = zeros(length(idxAccess) + size(UnlabeledFM, 1), size(TrainFM, 2));

    augFM(1:size(UnlabeledFM, 1), :) = UnlabeledFM(1:size(UnlabeledFM, 1), :);

    augFM( size(UnlabeledFM, 1) + 1 : size(UnlabeledFM, 1) + length(idxAccess), :) = TrainFM(idxAccess, :);

    Labels = zeros(size((augFM), 1), 1);
    
    Labels( size(UnlabeledFM, 1) + 1 : size(UnlabeledFM, 1) + length(idxAccess) ) = 1;
    
    % as of now, we can only use 
    [~, ~, newPositives{i}] = augmentData(augFM, Labels, [], 0, 0, 1, kkdThreshold, 10000000);
    
    newPositives{i} =  newPositives{i} (LabelToArray( newPositives{i} <= size(UnlabeledFM, 1) ))  ;
    
end

newPos = cell(numClasses);

for i = 1 : numClasses
    newPos{i} = newPositives{i};
end

for i = 1 : numClasses
    
    numMemb = size(newPositives{i})
    
    for j = 1 : numClasses
        
        if i ~= j
            
            newPositives{i} = setdiff( newPositives{i}, newPos{j});
            
        end        
    end    
end

for i = 1 : numClasses
        
    if ( ~isempty( newPositives{i} ) )
        
        str = ['Number of new positives of class', num2str(i), 'is:', num2str(size(newPositives{i}, 2))];
        disp(str)
        
        TrFM = zeros( size(TrainFM, 1) + size(newPositives{i}, 2), size(TrainFM, 2));
        TrFM(1:size(TrainFM, 1), :) = TrainFM( 1 : size(TrainFM, 1), :);
        TrFM( size(TrainFM, 1) + 1 : size(TrFM, 1), :) = UnlabeledFM( newPositives{i}, : );

        TrainL = zeros( size(TrainFM, 1) + size(newPositives{i}, 2), 1);
        TrainL(1:size(TrainFM, 1), :) = TrainLabels( 1 : size(TrainFM, 1));
        TrainL(size(TrainFM, 1) + 1 : size(TrFM, 1)) = i;

        TrainFM = TrFM;
        TrainLabels = TrainL;
    
    end
    
end

newLabelling = MultiClassSVM(TrainFM, TrainLabels, TestFM, 0);

augmentedResults = quality(newLabelling, TestLabels);
    