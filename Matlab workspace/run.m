function [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12] = run(TrainFM, TrainLabels, TestFM, TestLabels) 

[a1, a2, a3, a4]  = assessDataAugmentation(TrainFM, TrainLabels, TestFM, TestLabels, 0.1, 10000000, 1, 1, 1, 1, 1);
%We might want to do bidirectional pull with just removing intersection and
%not adding negatives at all???
%[ a5, a6, a7, a8]  = assessDataAugmentation(TrainFM, TrainLabels, TestFM, TestLabels, 0.5, 10000000, 1, 0, 1, 0, 0)

%[a9, a10, a11, a12]  = assessDataAugmentation(TrainFM, TrainLabels, TestFM, TestLabels, 0.9, 10000000, 1, 0, 1, 0 , 0)

a5 = 0; 
a6 = 0;
a7 = 0; 
a8 = 0;
a9 = 0;
a10 = 0;
a11 = 0;
a12 = 0;
