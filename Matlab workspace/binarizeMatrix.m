function binarizedMatrix = binarizeMatrix(Mat)

n = size(Mat, 1);
m = size(Mat, 2);

binarizedMatrix = zeros(n,m);

for i = 1:n
   binarizedMatrix(i,:) = binarizePosteriors(Mat(i,:), 0);
   i
end

