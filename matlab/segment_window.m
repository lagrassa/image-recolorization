function output = segment_window(orig)

window = roipoly(orig);
[row, col, ~] = size(orig);
result = zeros(row,col,3);
% imshow(window)
for i=1:row
    for j=1:col
        if window(i,j) == 1
            result(i,j,:) = orig(i,j,:);
        end
    end
end
output = result;
close all
end