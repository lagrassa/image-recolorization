function [output] = recolorize(input,img1)
partitions = double(max(input(:)));
c = zeros(partitions,3);
for i=1:partitions
    A = input == i;
    figure
    subplot(2,1,1)
    imshow(img1)
    subplot(2,1,2)
    imshow(A)
    c(i,1:3) = uisetcolor;
    close all
end
c_hsv = rgb2hsv(c);
img1_hsv = rgb2hsv(img1);
[row, col] = size(A);
recolor = zeros(size(img1));
for i=1:row
    for j=1:col
        for k=1:partitions
            if input(i,j) == k
                if sum(c_hsv(k,1:3)) == 0 % if black then use original color
                    recolor(i,j,1:3) = img1_hsv(i,j,1:3);
                else
                    recolor(i,j,1) = c_hsv(k,1); % use user-defined hue
                    recolor(i,j,2:3) = img1_hsv(i,j,2:3); % keep saturation + intensity the same
%                     
%                     recolor(i,j,1:2) = c_hsv(k,1:2); % use user-defined hue
%                     recolor(i,j,3) = img1_hsv(i,j,3); % keep saturation + intensity the same

%                     recolor(i,j,1:3) = c_hsv(k,1:3); % use user-defined HSV                    
                end
            end
        end
    end
end
output = hsv2rgb(recolor);
end
