function [init_1,init_2] = gen_unique_vals(password)
    ascii_pw_array = double(uint64(password));

    outputArg1 = 0.;
    outputArg2 = 0.;
    
    for i=ascii_pw_array
       outputArg1 = (outputArg1*31) + i;
       outputArg2 = (outputArg2*37) + i;
    end
    init_1 = outputArg1/double(intmax('uint64'));
    init_2 = outputArg2/double(intmax('uint64'));
end

