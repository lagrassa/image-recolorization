function output = run_kmeans_lab(I,k)
% Segment by HSV color space


% hsv = rgb2hsv(I);
% hsv(:,:,2) = hsv(:,:,2)*1.5;
% rgb = hsv2rgb(hsv);
% rgb = localcontrast(255*uint8(I));
lab = rgb2lab(I);

sz = size(lab);
im = single(reshape(lab,sz(1)*sz(2),[]));
im = im - mean(im);
im = im ./ std(im);

L = kmeans(im,k);
output = reshape(L,[sz(1) sz(2)]);

end