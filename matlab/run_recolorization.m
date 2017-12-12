%% run_recolorization using user-defined polygon
clear all;
img = imread('nnoutput.jpg');
window = segment_window(img); % polygon
while(true)
    k = input('Input k = the number of desired segments. A good default value is 3 \n Type quit to quit');
    if k == 'quit':
        break;
    output = run_kmeans_lab(window,k); % select k, default is 3
    close all
    [img_recolor] = recolorize(output,img);
    imshow(img_recolor)
end
