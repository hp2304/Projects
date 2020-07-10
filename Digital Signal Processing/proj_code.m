%% Setting the values of parameters and calculating all the angles between 0 degree to 45 degrees.

% Parameter N, num_bits, e_max can be changed as per preference.
clc;clear;close all;
tic
num_bits = 6;
N = 8;
angles=[];
for n = 0:N-1
    temp = (2*pi*n)/N;
    if temp<=(pi/4)
        angles = [angles, temp];
    end
end
e_max = 0.05;
range = 0:2^(num_bits-1)-1;
delta = asin(e_max); % Helps find the angle's upper and lower bounds.
%% Filtering out all the points based on the angles.
angle_filter = [];
for i = angles
    higher_bound = i + delta;   
    lower_bound = i - delta;
    for x = range
        for y = range
            phase = atan(y/x);
            if phase <= higher_bound && phase >= lower_bound
                angle_filter = [angle_filter; i x+j*y]; % store the point if the angle of the point is in the bounds
            end
        end
    end
end
%% Store all the points for respective angles in vectors.
cnt = 1;
for i = angles
    vec = [];
	vec = angle_filter(find(angle_filter(:,1)==i), 2);
    data{cnt} = vec;
    cnt = cnt+1;
end
%% Make kernel matrix by making every possible combinations of those vectors. 
kernel = combvec(data{1}.', data{2}.');
for j = 3:size(data,2)
     kernel = combvec(kernel, data{j}.');
end
kernel = kernel.';
clear angle_filter;
clear data;
%% For each row in kernel matrix calculate the error. 
min_error = 99;
error_idx = 1;
radius = 99;
for i = 1:size(kernel, 1)
   points = kernel(i, :);
   [error(i), rad] = find_error_from_points(points, angles); % Calculate error for each row.
   if(error(i) < min_error)
       min_error = error(i);
       error_idx = i;
       radius = rad;
   end
end
%% Print the radius, optimal coefficients (points) and minimum error.
radius
%kernel(find(error > e_max), :) = [];
%error(find(error > e_max)) = [];
min_error
kernel(error_idx,:)
toc