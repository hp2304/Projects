function [mapping] = get_map(list1,list2)
    list1 = uint16(list1);
    list2 = uint16(list2);
    
    len = length(list1);
    mapping = zeros(1,len*len);
    idx=1;
    
    for i=1:len
        for j=1:len
            mapping(idx) = bitor(bitshift(list1(j),8), list2(i));
            idx = idx+1;
        end
    end
end

