function [err, radius] = find_error_from_points(points, angles)
    for j = 1:length(points)
        re = real(points(j));
        im = imag(points(j));
        temp_r = abs(points(j));
        e_c = (re/temp_r) - cos(angles(j));
        e_s = (im/temp_r) - sin(angles(j));
        error(j) = sqrt((e_c)^2 + (e_s)^2);
    end
    [first_max_val, first_max_idx] = max(error);
    point_k = points(first_max_idx);
    error(first_max_idx) = -inf;
    [sec_max_val, sec_max_idx] = max(error);
    point_i = points(sec_max_idx);
    
    % C and S are respective cosine and sine components.
    c_i_part = real(point_i)*cos(angles(sec_max_idx));
    s_i_part = imag(point_i)*sin(angles(sec_max_idx));
    c_k_part = real(point_k)*cos(angles(first_max_idx));
    s_k_part = imag(point_k)*sin(angles(first_max_idx));
    radius = abs(point_i)^2 - abs(point_k)^2;
    radius = radius/(2*(c_i_part+s_i_part-c_k_part-s_k_part));
    radius;
    for j = 1:length(points)
        re = real(points(j));
        im = imag(points(j));
        e_c = (re/radius) - cos(angles(j));
        e_s = (im/radius) - sin(angles(j));
        error(j) = sqrt((e_c)^2 + (e_s)^2);
    end
    err = max(error);
    
end
